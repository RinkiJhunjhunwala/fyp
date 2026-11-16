import re, os

with open('thesis/build_fast_thesis.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract inlined_bibliography
start = text.find('inlined_bibliography = r"""') + len('inlined_bibliography = r"""')
end = text.find('"""\n\n# 3. Super fast')
inlined_bib = text[start:end].strip()

with open('thesis/inlined_bib.tex', 'w', encoding='utf-8') as f:
    f.write(inlined_bib)
print('Wrote thesis/inlined_bib.tex')

with open('thesis/thesis_nita.tex', 'r', encoding='utf-8') as f:
    tn = f.read()

# Replace \bibliographystyle and \bibliography with inlined_bib
old_bib = "\\bibliographystyle{plainnat}\n\\bibliography{references}"
if old_bib in tn:
    tn = tn.replace(old_bib, inlined_bib)
    with open('thesis/thesis_nita.tex', 'w', encoding='utf-8') as f:
        f.write(tn)
    print("Inlined bibliography into thesis_nita.tex successfully!")
else:
    print("Old bib pattern not found in thesis_nita.tex")
