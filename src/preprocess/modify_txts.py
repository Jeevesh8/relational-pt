for filename in ["test", "train", "valid"]:
    with open("subtrees-text-4096/" + filename + ".txt") as f:
        with open("subtrees-text-4096-64-comps/" + filename + ".txt", "w+") as g:
            for line in f.readlines():
                if line.count("<post") <= 64:
                    g.write(line)