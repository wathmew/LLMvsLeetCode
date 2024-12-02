import json

lc1 = "trap"

lc2 = "largestRectangleArea"

lc3 = "findMedianSortedArrays"

lc4 = "minWindow"

lc5 = "maxSlidingWindow"

lc6 = "mergeKLists"

lc7 = "reverseKGroup"

lc8 = "maxPathSum"

lc9 = "reversePairs"

lc10 = "findWords"

lc11 = "solveNQueens"

lc12 = "jobScheduling"

lc13 = "ladderLength"

lc14 = "minInterval"

lc15 = "findItinerary"

lc16 = "swimInWater"

lc17 = "longestIncreasingPath"

lc18 = "numDistinct"

lc19 = "maxCoins"

lc20 = "isMatch"

lc21 = "solveSudoku"

lc22 = "minCut"

lc23 = "maxProfit"

lc24 = "calculate"

lc25 = "removeInvalidParentheses"

lc26 = "maxCoins"

lc27 = "countSmaller"

lc28 = "maxNumber"

lc29 = "countRangeSum"

lc30 = "minPatches"

lc31 = "maxEnvelopes"

lc32 = "canCross"

lc33 = "strongPasswordChecker"

lc34 = "findMaximizedCapital"

lc35 = "findRotateSteps"

lc36 = "findMinMoves"

lc37 = "findIntegers"

lc38 = "scheduleCourse"

lc39 = "strangePrinter"

lc40 = "evaluate"

lc41 = "crackSafe"

lc42 = "slidingPuzzle"

lc43 = "hitBricks"

lc44 = "racecar"

lc45 = "mincostToHireWorkers"

lc46 = "nthMagicalNumber"

lc47 = "orderlyQueue"

lc48 = "movesToStamp"

lc49 = "oddEvenJumps"

lc50 = "maxSizeSlices"


lc_func = [globals()[f"lc{i}"] for i in range(1, 51)]

with open("lc_hard.json", "r") as file:
    data = json.load(file)

for i, entry in enumerate(data):
    entry["func"] = lc_func[i]

with open("lc_hard.json", "w") as file:
    json.dump(data, file, indent=4)

print("updated lc hard dataset function names")

