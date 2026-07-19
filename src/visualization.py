import matplotlib.pyplot as plt
import seaborn as sns
def plot_histograms(data):
    for column in data.columns[:-1]:
        plt.figure(figsize=(6,4))
        sns.histplot(data[column], kde=True)
        plt.title(f"Histogram of {column}")
        plt.tight_layout()
        plt.show()
def correlation_heatmap(data):
    plt.figure(figsize=(10,8))
    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
def boxplots(data):
    for column in data.columns[:-1]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.show()
def probability_chart(diabetic, normal):
    plt.figure(figsize=(5,5))
    plt.bar(
        ["Diabetic","Non-Diabetic"],
        [diabetic*100, normal*100]
    )
    plt.ylabel("Probability (%)")
    plt.title("Prediction Result")
    plt.tight_layout()
    plt.show()
