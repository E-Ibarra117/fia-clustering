import pprint   # For proper print of sequences.
import treetaggerwrapper
#1) build a TreeTagger wrapper:
tagger = treetaggerwrapper.TreeTagger(TAGLANG='es')


def doTagger(text):
    tags = tagger.tag_text(text)
    tags2 = treetaggerwrapper.make_tags(tags)
    return tags2


print("\n\n")

file1 = open('test-articulo.txt', 'r', encoding='utf-8')
Lines = file1.readlines()
  
result = []
for line in Lines:
    print(line.strip())
    result.append(doTagger(line.strip()))
    break

print(result)