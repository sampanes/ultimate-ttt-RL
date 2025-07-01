from engine.constants import EMPTY, X, O

# VERIFICATION TOOL
def verify_eq(actual, expected, label=""):
    if actual == expected:
        print(f"\033[92m✅ PASS\033[0m {label}")
        return True
    else:
        print(f"\033[91m❌ FAIL\033[0m {label}")
        print(f"    Expected: {repr(expected)}")
        print(f"    Got     : {repr(actual)}")
        return False
    
player_map = {None: "None",
              EMPTY: "None",
              X: "X",
              O: "O"}