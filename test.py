import pprint   # For proper print of sequences.
import treetaggerwrapper
#1) build a TreeTagger wrapper:
tagger = treetaggerwrapper.TreeTagger(TAGLANG='es')
#2) tag your text.
tags = tagger.tag_text("Ayer jugué futbol y ganamos 10 a 0")
#3) use the tags list... (list of string output from TreeTagger).

print("\n\n\n")

tags2 = treetaggerwrapper.make_tags(tags)
pprint.pprint(tags2)


# print(tags)