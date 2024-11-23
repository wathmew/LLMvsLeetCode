# LLMvsLeetCode

Notes:

any lc problems that involve creating some class rather than a function, just replace them; it'll make auto eval much easier

paste problem description in between the triple quotes for lc_hard_desc and then just run it to update the json

same with lc_hard_skel but put the function signature, I included any comments that the lc problem provided but if the leetcode problem uses multiline quotes, swap those to #

also same with lc_hard_refs. we'll have to figure out how to interpret the responses when we get them since we'll have to strip the newlines and such (maybe?)

I didn't finish all 25 because I took a break to watch the arcane finale, but ik fs that it works as intended so all we have to do is paste in the rest

I tested it with an additional statement for the other ref, but realistically idk if we're getting more than 1 for some of these because the other solutions are just like re-skins of the same thing
which is fine, and perhaps something we can talk about in the limitations section of the presentation

we'll still have to do test cases, which are probably easier to handle manually, but we'll have to figure out how to actually run the stuff with tests because I haven't thought about that much outside of us having to do dynamic code execution