import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv("IRIS.csv")

print(df.head())

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("🌸 Iris Dataset Dashboard", style={'textAlign': 'center'}),
    html.Br(),

    html.Label("Select Feature for Distribution:"),
    dcc.Dropdown(
        id="feature-dist",
        options=[{"label": col, "value": col} for col in df.columns if col != 'species'],
        value="sepal_width",
        clearable=False
    ),

    dcc.Graph(id="hist-plot"),

    html.Hr(),

    html.H2("Correlation Heatmap", style={'textAlign': 'center'}),
    dcc.Graph(
        id="corr-heatmap",
        figure=px.imshow(
            df.drop(columns=['species']).corr(),
            text_auto=True,
            color_continuous_scale="bluered",
            title="Feature Correlations"
        )
    )
])

@app.callback(
    Output("hist-plot", "figure"),
    Input("feature-dist", "value")
)
def update_hist(feature):
    fig = px.histogram(
        df,
        x=feature,
        color="species",
        barmode="overlay",
        nbins=20,
        title=f"Distribution of {feature}",
        template="plotly_white",
        opacity=0.7
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
