#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().with_name("apply_selective_int8_experiment.py")
text = p.read_text(encoding="utf-8")

old = '''activity = replace_once(
    activity,
    '        benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");\\n',
    '        benchmark.setText("Benchmark FP32 vs selective INT8");\\n',
    "benchmark label",
)
'''
new = '''benchmark_label = 'benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");'
benchmark_label_count = activity.count(benchmark_label)
if benchmark_label_count != 3:
    raise SystemExit(f"benchmark label: expected exactly three matches, found {benchmark_label_count}")
activity = activity.replace(benchmark_label, 'benchmark.setText("Benchmark FP32 vs selective INT8");')
'''
if old not in text:
    raise SystemExit("Selective INT8 patch benchmark-label block was not found")
text = text.replace(old, new, 1)

# The later generic replacement is now intentionally redundant; remove it so the
# patch script remains explicit about the three known UI restoration sites.
old_redundant = '''# The original benchmark label appears twice when the button is restored.
activity = activity.replace('benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");', 'benchmark.setText("Benchmark FP32 vs selective INT8");')
'''
text = text.replace(old_redundant, '', 1)

p.write_text(text, encoding="utf-8")
print("Selective INT8 patch anchors hardened successfully")
