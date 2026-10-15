from pathlib import Path
p=Path('build_report.py')
s=p.read_text(encoding='utf-8')
s=s.replace("g('sm_age.Age<30.n')", "g('sm_selfemp.Age<30.n')")
s=s.replace("g('sm_age.Age>=30.n')", "g('sm_selfemp.Age>=30.n')")
p.write_text(s,encoding='utf-8')
print('Patched build_report.py subgroup-size paths')
