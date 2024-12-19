## stim gate ids
# arbitrary 1q gate composed of elementary gates 
C1ids = [
    "I", "X", "Y", "Z",
    "H", "HX", "HY", "HZ",
    "S", "SX", "SY", "SZ",
    "HS", "HSX", "HSY", "HSZ",
    "SH", "SHX", "SHY", "SHZ",
    "HSH", "HSHX", "HSHY", "HSHZ"
]

Pauli_Twirls = ["I", "X", "Y", "Z",]

# Clifford rotations
RXids = ["I", "SQRT_X", "X", "SQRT_X_DAG"] # 0,pi/2,pi,3pi/2
RYids = ["I", "SQRT_Y", "Y", "SQRT_Y_DAG"]
RZids = ["I", "S", "Z", "S_DAG"]

# don't perform 2Q gate if None
Q2ids = [None, "CX", "CX", "SWAP"]