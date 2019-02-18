import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import flask
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

with open("../data/cand_df.pkl", 'rb') as table_file:
    df = pickle.load(table_file)


def generate_count(parties, areas, dataframe=df):
    dataframe = dataframe[dataframe.kunta.isin(areas) & dataframe.puolue_lyhyt.isin(parties)]

    return html.P('Candidates selected: '+str(len(dataframe)))

def generate_table(parties, areas, max_rows=5, dataframe=df):
    dataframe = dataframe[['kunta','puolue_lyhyt','sukunimi','etunimi','valittu',
                           'wiki_1','wiki_2', 'wiki_3', 'wiki_4', 'wiki_5']]
    dataframe = dataframe[dataframe.kunta.isin(areas) & dataframe.puolue_lyhyt.isin(parties)]
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def get_stacked_bar_for_party(stats, party):
    trace = go.Bar(
        x = stats.index,
        y = stats.values,
        name=party
    )
    return trace


def generate_count_figure(var, parties, areas, dataframe=df, min_limit=1):
    dataframe = dataframe[dataframe.kunta.isin(areas) & dataframe.puolue_lyhyt.isin(parties)]

    stats = dataframe.filter(regex="^"+var).melt().value.value_counts()
    category_order = stats.index
    return {
            #'data': [
            #    {'x': stats.index, 'y': stats.values, 'type': 'bar'},
            #],
            'data': [get_stacked_bar_for_party(dataframe[dataframe.puolue_lyhyt == party].filter(regex="^"+var).melt().
                                              groupby('value').filter(lambda x: len(x)>min_limit).
                                              value.value_counts()[category_order].fillna(0),
                                              party) for party in parties],
               #[get_stacked_bar_for_party(dataframe[dataframe.puolue_lyhyt == party].filter(regex="^"+var).melt().
               #                                groupby('value').filter(lambda x: len(x)>min_limit).
                #                               value.value_counts(), party) for party in parties],
            'layout': {
                'title': 'Wikipedia frequency chart',
                'barmode': 'stack'
            }
        }


kunta_dict = dict(zip(df.kunta, df.kunta))

party_dict ={'Suomen Keskusta': 'Kesk',
            'Kansallinen Kokoomus': 'Kok',
            'Suomen Sosialidemokraattinen Puolue':'SDP',
            'Vihreä liitto':'Vihr',
            'Vasemmistoliitto':'Vas',
            'Perussuomalaiset':'PS',
            'Suomen Kristillisdemokraatit (KD)':'KD',
            'Suomen ruotsalainen kansanpuolue':'RKP',
            'Suomen Kommunistinen Puolue':'SKP',
            'Piraattipuolue':'Pir',
            'Liberaalipuolue - Vapaus valita':'LP',
            'Feministinen puolue':'FP',
            'Itsenäisyyspuolue':'IP',
            'Muut': 'Muut'}


df[['wiki_1', 'wiki_2', 'wiki_3', 'wiki_4', 'wiki_5']] = df.wiki_closest.str.split('; ',-1, expand=True)
df['puolue_lyhyt'] = [party_dict[key] if key in party_dict else 'Muut' for key in df.puolue ]


app.layout = html.Div(children=[
    html.H1(children='Kuntavaalit-Wikipedia LDA'),

    html.Div(children='''
        Mikä Wikipedia-sivu vastaa parhaiten ehdokkaan tekstivastauksia?
    '''),

    html.Label('Select parties:'),
    dcc.Dropdown(
        id='party_dropdown',
        options=[{'label': party_dict.get(key), 'value': party_dict.get(key)} for key in party_dict.keys()],
        value=list(party_dict.values()),
        multi=True
    ),
    html.Label('Select areas:'),
    dcc.Dropdown(
        id="area_dropdown",
        options=[{'label': key, 'value': kunta_dict.get(key)} for key in kunta_dict.keys()],
        value=['Helsinki', 'Espoo'],
        multi=True
    ),
    html.Button('Select All', id='select-all'),
    html.Label('Select top wiki variables:'),
    dcc.RadioItems(
        id='variable_selector',
        options=[{'label': 'Wiki 1', 'value': 'wiki_1'},
        {'label': 'Wiki 2', 'value': 'wiki_2'},
        {'label': 'Wiki 3', 'value': 'wiki_3'},
        {'label': 'Wiki 4', 'value': 'wiki_4'},
        {'label': 'Wiki 5', 'value': 'wiki_5'},
        {'label': 'All', 'value': 'wiki_[1-5]'}
        ],
        value="wiki_1",
        labelStyle={'display': 'inline-block'}
    ),

    html.Div(id='count_container'),

    dcc.Graph(
        id='count-graph'
    ),
    html.Div(id='output_container')



])


@app.callback(
    dash.dependencies.Output('output_container', 'children'),
    [dash.dependencies.Input('party_dropdown', 'value'),
     dash.dependencies.Input('area_dropdown', 'value')])
def update_output(parties, areas):
    return generate_table(parties, areas)

@app.callback(
    dash.dependencies.Output('count-graph', 'figure'),
    [dash.dependencies.Input('variable_selector','value'),
     dash.dependencies.Input('party_dropdown', 'value'),
     dash.dependencies.Input('area_dropdown', 'value')
     ])
def update_count_figure(selected_wiki_variable, parties, areas):
    return generate_count_figure(selected_wiki_variable, parties, areas)

@app.callback(
    dash.dependencies.Output('count_container', 'children'),
    [dash.dependencies.Input('party_dropdown', 'value'),
     dash.dependencies.Input('area_dropdown', 'value')])
def update_count(parties, areas):
    return generate_count(parties, areas)

@app.callback(
    dash.dependencies.Output('area_dropdown','value'),
    [dash.dependencies.Input('select-all', 'n_clicks')],
    [dash.dependencies.State('area_dropdown', 'options'),
     dash.dependencies.State('area_dropdown', 'value')]
)
def select_all_areas(n_clicks, options, values):
    if n_clicks is not None and n_clicks > 0:
        n_clicks = 0
        return [i['value'] for i in options]
    else:
        return values


#if __name__ == '__main__':
#    app.run_server(debug=False, port=8050)

app.run_server(host='0.0.0.0', debug=False, port=8050)
