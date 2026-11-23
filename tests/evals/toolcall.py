"""Multi-turn + tool-call smoke, and a reasoning-effort sweep.
Tool chat: user asks for weather -> assistant must emit a get_weather tool call (structure + city arg checked) ->
we send a tool result with an unusual value -> final answer must use it -> follow-up turn must convert it (73.4 F).
Effort sweep: one word problem at none/low/medium/high; reasoning length should grow with effort (info, not gate).
Usage: python tests/evals/toolcall.py [--base URL]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import main_for  # noqa: E402

TOOLS = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Current weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
RESULT = {'city': 'Reykjavik', 'temp_c': 23, 'condition': 'record heatwave', 'station': 'WX-5517'}
SWEEP_Q = ('A train leaves at 9:40 and travels 237 km at 79 km/h, then waits 25 minutes, then travels 156 km at '
           '104 km/h. At what time does it arrive? Give the final answer as "Answer: HH:MM".')
SWEEP_A = '14:35'  # 9:40 + 3 h + 25 min + 1.5 h


def tool_chat(client, effort='low'):
    checks, msgs = {}, [{'role': 'user', 'content': 'What is the weather in Reykjavik right now? Use the tool.'}]
    r1 = client.chat(msgs, tools=TOOLS, effort=effort, max_tokens=2048)
    tc = (r1['tool_calls'] or [{}])[0]
    fn = tc.get('function') or {}
    try:
        args = json.loads(fn.get('arguments') or 'null')
    except ValueError:
        args = None
    checks['t1_finish_tool_calls'] = r1['finish'] == 'tool_calls'
    checks['t1_tool_call_shape'] = bool(tc.get('id')) and tc.get('type') == 'function' and fn.get('name') == 'get_weather'
    checks['t1_args_json_city'] = isinstance(args, dict) and 'reykjav' in str(args.get('city', '')).lower()
    call = {'id': tc.get('id') or 'call_0', 'type': 'function',
            'function': {'name': 'get_weather', 'arguments': json.dumps({'city': 'Reykjavik'})}}
    msgs += [{'role': 'assistant', 'content': r1['content'] or None, 'tool_calls': [call]},
             {'role': 'tool', 'tool_call_id': call['id'], 'name': 'get_weather', 'content': json.dumps(RESULT)}]
    r2 = client.chat(msgs, tools=TOOLS, effort=effort, max_tokens=2048)
    checks['t2_uses_result'] = '23' in r2['content'] and 'heat' in r2['content'].lower()
    checks['t2_no_new_tool_call'] = not r2['tool_calls']
    msgs += [{'role': 'assistant', 'content': r2['content']},
             {'role': 'user', 'content': 'Convert that temperature to Fahrenheit. Reply with just the number.'}]
    r3 = client.chat(msgs, tools=TOOLS, effort=effort, max_tokens=2048)
    checks['t3_followup_73'] = '73' in r3['content']
    turns = [dict(content=r['content'][:300], tool_calls=r['tool_calls'], finish=r['finish'], error=r['error'])
             for r in (r1, r2, r3)]
    return checks, turns


def sweep(client, levels=('none', 'low', 'medium', 'high')):
    rows = []
    for lv in levels:
        r = client.chat([{'role': 'user', 'content': SWEEP_Q}], effort='off' if lv == 'none' else lv, max_tokens=12000)
        rows.append(dict(effort=lv, reasoning_chars=len(r['reasoning']), completion_tokens=r['usage'].get('completion_tokens'),
                         correct=SWEEP_A in r['content'], finish=r['finish'], secs=round(r['secs'], 1), error=r['error']))
        print(f"effort {lv}: reasoning {rows[-1]['reasoning_chars']} chars, {rows[-1]['completion_tokens']} tokens, "
              f"correct {rows[-1]['correct']}, {r['secs']:.0f}s {r['error'] or ''}", flush=True)
    return rows


def run(client, quick=False):
    checks, turns = tool_chat(client)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    rows = sweep(client)
    lens = [r['reasoning_chars'] for r in rows if not r['error']]
    summary = {'tool_checks_passed': sum(checks.values()), 'tool_checks_total': len(checks),
               'effort_monotonic': all(a <= b for a, b in zip(lens, lens[1:])) and len(lens) == len(rows)}
    for r in rows:
        summary[f"effort_{r['effort']}_reasoning_chars"] = r['reasoning_chars']
        summary[f"effort_{r['effort']}_correct"] = r['correct']
    return dict(summary=summary, details=dict(checks=checks, turns=turns, sweep=rows))


if __name__ == '__main__':
    main_for(run, __doc__)
