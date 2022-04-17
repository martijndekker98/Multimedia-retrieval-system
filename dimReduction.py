from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
import step4
import pandas
import plotly.express as px


# Given a dataframe, constructs the dimensionality reduction graph using tsne
def testTSNE(dataframe, useplt: bool = False):
    data = dataframe.iloc[:, 2:]
    print(data)
    setLabels = list(dataframe["Subfolder"].unique())
    dataY = dataframe.copy()
    dataY['y'] = dataY.apply(lambda row: row.iloc[0], axis=1)
    tsne = manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=4000)
    tsne_results = tsne.fit_transform(data)
    print("Done fitting")

    tsne_x = "tsne-x"
    tsne_y = "tsne-y"
    if useplt:
        dataY[tsne_x] = tsne_results[:,0]
        dataY[tsne_y] = tsne_results[:,1]
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x=tsne_x, y=tsne_y,
            hue="y",
            palette=sns.color_palette("hls", len(setLabels)),
            data=dataY,
            legend="full",
            alpha=0.5)
        plt.show()
    else:
        colours = ['#2e8b57', '#808000', '#ff1493', '#d3d3d3', '#00009b', '#ff0000', '#ff8c00', '#ba55d3', '#fa8072', '#00ff7f',
                   '#00ffff', '#8b0000', '#0000ff', '#adff2f', '#ff00ff', '#00bfff', '#f0e68c', '#dda0dd', '#ffd700']
        nieuwDF = pandas.DataFrame()
        nieuwDF[tsne_x] = tsne_results[:,0]
        nieuwDF[tsne_y] = tsne_results[:,1]
        nieuwDF['label'] = dataframe.iloc[:,0]
        lijst = [x[:-4] for x in dataframe.iloc[:,1]]
        nieuwDF['ID'] = lijst
        # fig = px.scatter(nieuwDF, x=tsne_x, y=tsne_y, text="ID", color="label", hover_data=['label'])
        fig = px.scatter(nieuwDF, x=tsne_x, y=tsne_y, text="ID", color="label", hover_data=['label', 'ID'], color_discrete_sequence=colours)
        fig.update_traces(textposition='top center')
        fig.show()


if __name__ == '__main__':
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    testTSNE(dataframe)

