def compare_protein_sequences(seq1, seq2):
    # Ensure sequences are of the same length
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length to compare")

    # Iterate over the sequences and find differences
    differences = []
    for i, (res1, res2) in enumerate(zip(seq1, seq2)):
        if res1 != res2:
            differences.append((i, res1, res2))

    pos = []
    mut = []

    # Print the differences
    if not differences:
        print("The sequences are identical.")
    else:
        # print(f"Differences found at {len(differences)} positions:")
        for i, res1, res2 in differences:
            print(f"Position {i+1}:\tWildtype: {seq1[i]} -> Mutant: {seq2[i]}")
            pos.append(i+1)
            mut.append(seq2[i])

    # print(pos)
    # print(mut)

# Example usage
seq1 = "LAAVSVDCSEYPKPACTLEYRPLCGSDNKTYGNKCNFCNAVVESNGTLTLSHFGKC"
seq2 = "LAAVSVDCSEYPKPACTDEYRPLCGSDNKTYGNKCNFCNAVVESNGTLTLSHFGKC"
compare_protein_sequences(seq1, seq2)


