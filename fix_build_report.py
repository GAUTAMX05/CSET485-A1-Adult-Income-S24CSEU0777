from pathlib import Path
p=Path('build_report.py')
s=p.read_text(encoding='utf-8')
s=s.replace("g('sm_age.Age<30.n')", "g('sm_selfemp.Age<30.n')")
s=s.replace("g('sm_age.Age>=30.n')", "g('sm_selfemp.Age>=30.n')")
s=s.replace("They encode different decision policies.", "They encode different decision policies. That distinction matters because a score is not itself a policy. The analyst still chooses the cut point, decides which errors are acceptable, and determines whether a person receives review or an automatic outcome.")
p.write_text(s,encoding='utf-8')
print('Patched build_report.py subgroup-size paths and Q1 word count')
