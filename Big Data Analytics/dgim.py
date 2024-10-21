import math

class DGIM:
    def __init__(self, window_size):
        # Set the window size N
        self.window_size = window_size
        # List of buckets, each bucket is a tuple (timestamp, size)
        self.buckets = []
    
    def add_bit(self, bit, timestamp):
        # Only care about 1's in the stream
        if bit == 1:
            # Add a new bucket for the current bit with size 1
            self.buckets.insert(0, (timestamp, 1))
            
            # Merge buckets if there are more than two of the same size
            self._merge_buckets()

        # Remove old buckets that are out of the sliding window
        self._expire_old_buckets(timestamp)

    def _merge_buckets(self):
        # Merge buckets so that no more than two buckets of the same size exist
        i = 0
        while i < len(self.buckets) - 2:
            # If three consecutive buckets have the same size, merge the oldest two
            if self.buckets[i][1] == self.buckets[i+1][1] == self.buckets[i+2][1]:
                # Merge the two oldest buckets (i+1 and i+2)
                new_bucket = (self.buckets[i+1][0], self.buckets[i+1][1] * 2)
                # Remove the two old buckets and insert the new one
                del self.buckets[i+1:i+3]
                self.buckets.insert(i+1, new_bucket)
            else:
                i += 1

    def _expire_old_buckets(self, current_time):
        # Remove buckets that are out of the current window size
        while self.buckets and self.buckets[-1][0] <= current_time - self.window_size:
            self.buckets.pop()

    def count_ones(self, current_time):
        # Count the number of 1's in the last N bits (sliding window)
        total = 0
        for i, (timestamp, size) in enumerate(self.buckets):
            if timestamp <= current_time - self.window_size:
                break
            if i == len(self.buckets) - 1:
                # The oldest bucket might not fully overlap with the window
                # Add only half of its size for the approximation
                total += size // 2
            else:
                total += size
        return total

    def display_buckets(self):
        # Display the current state of the buckets
        print("Final Buckets (timestamp, size):", self.buckets)


# Example Usage
window_size = 24
dgim = DGIM(window_size)

# Stream of bits arriving over time
stream = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

# Process the stream
for t, bit in enumerate(stream):
    dgim.add_bit(bit, t)

# At the last bit, count the number of 1's and display the final state of the buckets
ones_count = dgim.count_ones(len(stream) - 1)
print(f"\nAt the last bit, number of 1's in the last {window_size} bits: {ones_count}")
dgim.display_buckets()  
