import re
urec = []
with open('/tmp/v18_eval.log') as f:
    for l in f:
        m = re.search(r'U_recall=([\d.]+)', l)
        if m: urec.append(float(m.group(1)))
n = len(urec)
print(f'Total rounds: {n}')
sess = {i: [] for i in range(4)}
for idx, val in enumerate(urec):
    sess[idx % 4].append(val)
for s in range(4):
    v = sess[s]
    print(f'S{s+1}: mean={sum(v)/len(v):.3f}  min={min(v):.2f}  max={max(v):.2f}')
print(f'Overall U_recall: {sum(urec)/len(urec):.3f}')
