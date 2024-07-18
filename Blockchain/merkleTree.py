import hashlib

class MerkleNode: 
    def __init__(self, hash, value = None):
        self.hash = hash
        self.value = value # for printing transaction values of leave nodes (not important) 
        self.left = None
        self.right = None

def merkleTree(transactions):
    # exit if no transactions
    if len(transactions) == 0: 
        return None
    # create node for each transaction
    nodes = [MerkleNode(hashlib.sha256(transaction.encode()).hexdigest(), transaction) for transaction in transactions]
    # create the binary tree
    # traverse nodes until one root node remains
    while len(nodes) > 1: 
        newLevel = []
        for i in range(0, len(nodes), 2):
            # create combined hash of left and right nodes (repeat last one if odd)
            left = nodes[i]
            right = nodes[i+1] if i+1 < len(nodes) else nodes[i]
            combinedHash = hashlib.sha256((left.hash + right.hash).encode()).hexdigest()
            # create parent node
            parent = MerkleNode(combinedHash)
            parent.left = left
            parent.right = right
            # add parent in current level
            newLevel.append(parent)
        # update the nodes for next iteration
        nodes = newLevel
    return nodes[0]

# print the merkle tree recursively
def printMerkleTree(merkleRoot, root):
    if root: print("\nRoot ", end="")
    print("Hash: " + merkleRoot.hash)
    if merkleRoot.right: print("Left: " + merkleRoot.left.hash)
    if merkleRoot.right: print("Right: " + merkleRoot.right.hash)
    if merkleRoot.value: print("Value: " + merkleRoot.value)
    print()
    if (merkleRoot.left == None or merkleRoot.right == None):
        return
    else: 
        printMerkleTree(merkleRoot.left, False)
        printMerkleTree(merkleRoot.right, False)


transactions = ["t1", "t2", "t3", "t4"]
merkleRoot = merkleTree(transactions)
printMerkleTree(merkleRoot, True)