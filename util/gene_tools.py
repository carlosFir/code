import matplotlib.pyplot as plt


def DrawPlot(out_concat_values, seqs_values):
        
    out_concat_values = out_concat_values[0].cpu()
    seqs_values = seqs_values[0].cpu()

    x = range(len(out_concat_values))
    plt.figure(figsize=(10,5))
    plt.plot(x, out_concat_values, label='out_concat')
    plt.plot(x, seqs_values, label='seqs')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig('cts.png')