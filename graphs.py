import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

metrics = [
    "Genre",
    "Accuracy",
    "Loss",
    "F1",
    "Precision",
    "Recall",
    "BAC Score",
    "Specificity",
    "GMean",
    "Hamming",
    "0/1",
]
bar_width = 0.25
genres = [
    "Ambient",
    "Classical",
    "Dance",
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Industrial & Noise",
    "Jazz",
    "Metal",
    "Pop",
    "Psychedelia",
    "Punk",
    "R&B",
    "Rock",
    "Singer-Songwriter",
]

matplotlib.rcParams["figure.dpi"] = 600
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


def experiment_comparison_plot():
    df1_0 = pd.read_csv("./Results/results_experiment_1_Spectrograms.csv")
    df1_1 = pd.read_csv("./Results/results_experiment_1_MelSpectrograms.csv")

    df2_0 = pd.read_csv("./Results/results_experiment_2_Spectrograms.csv")
    df2_1 = pd.read_csv("./Results/results_experiment_2_MelSpectrograms.csv")

    df3_0 = pd.read_csv("./Results/results_experiment_3_Spectrograms.csv")
    df3_1 = pd.read_csv("./Results/results_experiment_3_MelSpectrograms.csv")

    df4_0 = pd.read_csv("./Results/results_experiment_4_Spectrograms.csv")
    df4_1 = pd.read_csv("./Results/results_experiment_4_MelSpectrograms.csv")

    df5_0 = pd.read_csv("./Results/results_experiment_5_Spectrograms.csv")
    df5_1 = pd.read_csv("./Results/results_experiment_5_MelSpectrograms.csv")

    spec_acc = [
        df1_0["Accuracy"][16],
        df2_0["Accuracy"][16],
        df3_0["Accuracy"][16],
        df4_0["Accuracy"][16],
        df5_0["Accuracy"][16],
    ]
    mel_spec_acc = [
        df1_1["Accuracy"][16],
        df2_1["Accuracy"][16],
        df3_1["Accuracy"][16],
        df4_1["Accuracy"][16],
        df5_1["Accuracy"][16],
    ]

    plt.figure(figsize=(10, 10))
    plt.bar(
        [0.9, 1.9, 2.9, 3.9, 4.9],
        spec_acc,
        bar_width,
        label="Spectrograms",
    )
    plt.bar(
        [1.15, 2.15, 3.15, 4.15, 5.15],
        mel_spec_acc,
        bar_width,
        label="MelSpectrograms",
    )
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.axhline(y=0.5, color="r", linestyle="dashed")
    plt.ylim(bottom=0.45)
    plt.xticks(range(1, 6))  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("./Graphs/comparison_bar_mel.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.bar(
        [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
        df1_0.iloc[16, 3:],
        0.2,
        label="Experiment 1",
    )
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], df2_0.iloc[16, 3:], 0.2, label="Experiment 2")
    plt.bar(
        [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
        df3_0.iloc[16, 3:],
        0.2,
        label="Experiment 3",
    )
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.xticks(ticks=np.arange(1, 9), labels=metrics[3:])  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("./Graphs/comparison_bar_ex_123.png", bbox_inches="tight")
    plt.clf()


def stratified_comparison_metrics():
    df3_0 = pd.read_csv("./Results/results_experiment_3_Spectrograms.csv")
    df3_1 = pd.read_csv("./Results/results_experiment_3_MelSpectrograms.csv")

    df4_0 = pd.read_csv("./Results/results_experiment_4_Spectrograms.csv")
    df4_1 = pd.read_csv("./Results/results_experiment_4_MelSpectrograms.csv")

    df5_0 = pd.read_csv("./Results/results_experiment_5_Spectrograms.csv")
    df5_1 = pd.read_csv("./Results/results_experiment_5_MelSpectrograms.csv")

    plt.figure(figsize=(10, 10))
    plt.bar(
        [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
        df3_0.iloc[16, 3:],
        0.2,
        label="Experiment 3",
    )
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], df4_0.iloc[16, 3:], 0.2, label="Experiment 4")
    plt.bar(
        [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
        df5_0.iloc[16, 3:],
        0.2,
        label="Experiment 5",
    )
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.xticks(ticks=np.arange(1, 9), labels=metrics[3:])  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("./Graphs/comparison_bar_ex_345.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.bar(
        [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
        df3_1.iloc[16, 3:],
        0.2,
        label="Experiment 3",
    )
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], df4_1.iloc[16, 3:], 0.2, label="Experiment 4")
    plt.bar(
        [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
        df5_1.iloc[16, 3:],
        0.2,
        label="Experiment 5",
    )
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.xticks(ticks=np.arange(1, 9), labels=metrics[3:])  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("./Graphs/comparison_bar_ex_345_mel.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(0.8, 17.8, 1), df3_1["Accuracy"], 0.2, label="Experiment 3")
    plt.bar(np.arange(1, 18, 1), df4_1["Accuracy"], 0.2, label="Experiment 4")
    plt.bar(np.arange(1.2, 18.2, 1), df5_1["Accuracy"], 0.2, label="Experiment 5")
    plt.xlabel("Genre")
    plt.ylabel("Accuracy")
    plt.xticks(
        ticks=np.arange(1, 18), labels=df3_0["Genre"], rotation=90
    )  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.ylim(bottom=0.15)
    plt.savefig("./Graphs/comparison_ex_345_accuracy.png", bbox_inches="tight")
    plt.clf()


def recall_specificity_bac():
    df1_0 = pd.read_csv("./Results/results_experiment_1_Spectrograms.csv")
    df1_1 = pd.read_csv("./Results/results_experiment_1_MelSpectrograms.csv")

    df2_0 = pd.read_csv("./Results/results_experiment_2_Spectrograms.csv")
    df2_1 = pd.read_csv("./Results/results_experiment_2_MelSpectrograms.csv")

    df3_0 = pd.read_csv("./Results/results_experiment_3_Spectrograms.csv")
    df3_1 = pd.read_csv("./Results/results_experiment_3_MelSpectrograms.csv")

    df4_0 = pd.read_csv("./Results/results_experiment_4_Spectrograms.csv")
    df4_1 = pd.read_csv("./Results/results_experiment_4_MelSpectrograms.csv")

    df5_0 = pd.read_csv("./Results/results_experiment_5_Spectrograms.csv")
    df5_1 = pd.read_csv("./Results/results_experiment_5_MelSpectrograms.csv")

    for i, df in enumerate(
        [[df1_0, df1_1], [df2_0, df2_1], [df3_0, df3_1], [df4_0, df4_1], [df5_0, df5_1]]
    ):
        plt.figure(figsize=(10, 10))
        plt.bar(np.arange(0.8, 17.8, 1), df[0]["Recall"], 0.2, label="Recall")
        plt.bar(np.arange(1, 18, 1), df[0]["Specificity"], 0.2, label="Specificity")
        plt.bar(
            np.arange(1.2, 18.2, 1), df[0]["Balanced Accuracy"], 0.2, label="BAC Score"
        )
        plt.xlabel("Genre")
        plt.ylabel("Value")
        plt.xticks(
            ticks=np.arange(1, 18), labels=df[0]["Genre"], rotation=90
        )  # Set X-axis ticks
        plt.legend()
        plt.grid(axis="y")
        plt.ylim(bottom=0.1)
        plt.savefig("./Graphs/comparison_rsb_{}.png".format(i + 1), bbox_inches="tight")
        plt.clf()

        plt.figure(figsize=(10, 10))
        plt.bar(np.arange(0.8, 17.8, 1), df[1]["Recall"], 0.2, label="Recall")
        plt.bar(np.arange(1, 18, 1), df[1]["Specificity"], 0.2, label="Specificity")
        plt.bar(
            np.arange(1.2, 18.2, 1), df[1]["Balanced Accuracy"], 0.2, label="BAC Score"
        )
        plt.xlabel("Genre")
        plt.ylabel("Value")
        plt.xticks(
            ticks=np.arange(1, 18), labels=df[1]["Genre"], rotation=90
        )  # Set X-axis ticks
        plt.legend()
        plt.grid(axis="y")
        plt.ylim(bottom=0.1)
        plt.savefig(
            "./Graphs/comparison_rsb_mel_{}.png".format(i + 1), bbox_inches="tight"
        )
        plt.clf()


def basic_graphs():
    df1_0 = pd.read_csv("./Results/results_experiment_1_Spectrograms.csv")
    df1_1 = pd.read_csv("./Results/results_experiment_1_MelSpectrograms.csv")

    df2_0 = pd.read_csv("./Results/results_experiment_2_Spectrograms.csv")
    df2_1 = pd.read_csv("./Results/results_experiment_2_MelSpectrograms.csv")

    df3_0 = pd.read_csv("./Results/results_experiment_3_Spectrograms.csv")
    df3_1 = pd.read_csv("./Results/results_experiment_3_MelSpectrograms.csv")

    df4_0 = pd.read_csv("./Results/results_experiment_4_Spectrograms.csv")
    df4_1 = pd.read_csv("./Results/results_experiment_4_MelSpectrograms.csv")

    df5_0 = pd.read_csv("./Results/results_experiment_5_Spectrograms.csv")
    df5_1 = pd.read_csv("./Results/results_experiment_5_MelSpectrograms.csv")

    for i, df in enumerate(
        [[df1_0, df1_1], [df2_0, df2_1], [df3_0, df3_1], [df4_0, df4_1], [df5_0, df5_1]]
    ):
        plt.figure(figsize=(10, 10))
        plt.bar(
            [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9],
            df[0].iloc[16, 3:],
            bar_width,
            label="Spectrograms",
        )
        plt.bar(
            [1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15],
            df[1].iloc[16, 3:],
            bar_width,
            label="MelSpectrograms",
        )
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.xticks(ticks=np.arange(1, 9), labels=metrics[3:])  # Set X-axis ticks
        plt.legend()
        plt.grid(axis="y")
        plt.savefig(
            "./Graphs/comparison_ex_{}_metrics.png".format(i + 1), bbox_inches="tight"
        )
        plt.clf()

        plt.figure(figsize=(10, 10))
        plt.bar(
            np.arange(0.9, 17.9, 1), df[0]["Accuracy"], bar_width, label="Spectrograms"
        )
        plt.bar(
            np.arange(1.15, 18.15, 1),
            df[1]["Accuracy"],
            bar_width,
            label="MelSpectrograms",
        )
        plt.xlabel("Genre")
        plt.ylabel("Accuracy")
        plt.xticks(
            ticks=np.arange(1, 18), labels=df[0]["Genre"], rotation=90
        )  # Set X-axis ticks
        plt.legend()
        plt.grid(axis="y")
        plt.ylim(bottom=0.2)
        plt.savefig(
            "./Graphs/comparison_ex_{}_accuracy.png".format(i + 1), bbox_inches="tight"
        )
        plt.clf()


if __name__ == "__main__":
    # experiment_comparison_plot()
    # stratified_comparison_metrics()
    # recall_specificity_bac()
    # basic_graphs()
    pass
