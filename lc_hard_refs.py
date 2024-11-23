import json

lc1 = [
"""
def trap(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        water = 0

        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                water += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water += right_max - height[right]
        
        return water
"""
]

lc2 = [
"""

"""
]

lc3 = [
"""

"""
]

lc4 = [
"""

"""
]

lc5 = [
"""

"""
]

lc6 = [
"""

"""
]

lc7 = [
"""

"""
]

lc8 = [
"""

"""
]

lc9 = [
"""

"""
]

lc10 = [
"""

"""
]

lc11 = [
"""

"""
]

lc12 = [
"""

"""
]

lc13 = [
"""

"""
]

lc14 = [
"""

"""
]

lc15 = [
"""

"""
]

lc16 = [
"""

"""
]

lc17 = [
"""

"""
]

lc18 = [
"""

"""
]

lc19 = [
"""

"""
]

lc20 = [
"""

"""
]

lc21 = [
"""

"""
]

lc22 = [
"""

"""
]

lc23 = [
"""

"""
]

lc24 = [
"""

"""
]

lc25 = [
"""

"""
]

lc26 = [
"""

"""
]

lc27 = [
"""

"""
]

lc28 = [
"""

"""
]

lc29 = [
"""

"""
]

lc30 = [
"""

"""
]

lc31 = [
"""

"""
]

lc32 = [
"""

"""
]

lc33 = [
"""

"""
]

lc34 = [
"""

"""
]

lc35 = [
"""

"""
]

lc36 = [
"""

"""
]

lc37 = [
"""

"""
]

lc38 = [
"""

"""
]

lc39 = [
"""

"""
]

lc40 = [
"""

"""
]

lc41 = [
"""

"""
]

lc42 = [
"""

"""
]

lc43 = [
"""

"""
]

lc44 = [
"""

"""
]

lc45 = [
"""

"""
]

lc46 = [
"""

"""
]

lc47 = [
"""

"""
]

lc48 = [
"""

"""
]

lc49 = [
"""

"""
]

lc50 = [
"""

"""
]

lc_refs = [globals()[f"lc{i}"] for i in range(1, 51)]

with open("lc_hard.json", "r") as file:
    data = json.load(file)

def convert_to_json_string(multiline_string):
    return multiline_string.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')

lc_refs_single = [[convert_to_json_string(ref_string) for ref_string in lc_ref]  for lc_ref in lc_refs]

for i, entry in enumerate(data):
    if "ref" in entry:
        entry["ref"] = []
        refs = lc_refs_single[i]
        for ref in refs:
            entry["ref"].append(ref)
    else:
        print(f"json ref formatted incorrectly on entry {i}")
        break

with open("lc_hard.json", "w") as file:
    json.dump(data, file, indent=4)

print("updated lc hard dataset references")