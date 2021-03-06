import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import geopandas
from collections import defaultdict

with st.echo(code_location='below'):
    # cleaned version of dataset
    PATH_TO_DATASET = 'https://raw.githubusercontent.com/ParmenidesSartre/Suicide-Rates-Overview-1985-to-2016/master/master.csv'

    @st.cache
    def download_dataset():
        data = pd.read_csv(PATH_TO_DATASET)
        return data

    @st.cache
    def make_aggregated_data(data):
        """returns data aggregated by country and year, without segregation by generation and other parameters"""
        res = defaultdict(lambda: defaultdict(dict))
        for index, row in data.iterrows():
            year = int(row['year'])
            country = row['country']
            gdp_per_year = row[data.columns[9]]
            population = row['population']
            res[country][year]['gdp'] = int(gdp_per_year.replace(",",''))
            res[country][year]['population'] = res[country][year].get('population', 0) + population
            res[country][year]['n_suicides'] = res[country][year].get('suicides_no', 0) + row['suicides_no']
            res[country][year]['hdi'] = row['HDI for year']
        rows = []
        for country, year_data in res.items():
            for year, data in year_data.items():
                new_element = {'country': country, 'year': year, **data}
                rows.append(new_element)
        df = pd.DataFrame(rows)
        df['suicides_by_100k'] = df['n_suicides']*100000/df['population']
        df['gdp_per_capita'] = df['gdp']/df['population']
        return df
    
    ### FROM: https://discuss.streamlit.io/t/multiple-tabs-in-streamlit/1100/18
    def tabs(default_tabs = [], default_active_tab=0):
        """streamlit tabs from https://discuss.streamlit.io/t/multiple-tabs-in-streamlit/1100/18"""
        if not default_tabs:
            return None
        active_tab = st.radio("", default_tabs, index=default_active_tab)
        child = default_tabs.index(active_tab)+1
        st.markdown("""  
            <style type="text/css">
            div[role=radiogroup] > label > div:first-of-type {
               display: none
            }
            div[role=radiogroup] {
                flex-direction: unset
            }
            div[role=radiogroup] label {             
                border: 1px solid #999;
                background: #EEE;
                padding: 4px 12px;
                border-radius: 4px 4px 0 0;
                position: relative;
                top: 1px;
                }
            div[role=radiogroup] label:nth-child(""" + str(child) + """) {    
                background: #FFF !important;
                border-bottom: 1px solid transparent;
            }            
            </style>
        """,unsafe_allow_html=True)        
        return active_tab
    ### ENDFROM

    ru_columns = {
        'country': '????????????',
        'year': '?????? ????????????????????',
        'sex': '??????',
        'age': '??????????????',
        'suicides_no': '???????????????????? ??????????????????????',
        'population': '??????????????????',
        'suicides/100k pop': '?????????? ?????????????????????? ???? 100 ??????. ??????.',
        'country-year': '???????????????? ???????????? + ??????',
        'HDI for year': '???????????? ?????????????????????????? ???????????????????? ???? ??????',
        'gdp_for_year ($)': '?????????????? ??????',
        "gdp_per_capita ($)": '?????? ???? ???????? ??????????????????',
        'generation': '??????????????????'
    }

    st.sidebar.header("?????????? ?????????????????????? ?????????????????????? c 1985 ???? 2016 ??")
    data = download_dataset()
    tabs = [
        '???????????????? ???????????? ????????????', 
        "???????????????? ?????????????????? ??????????????????????", 
        "?????????????????????????? ???????????? ???? ??????????????", 
        "?????? ?????????? ?? ???????????????????????? ?????????????????????? ????????????????????",
        "???????????? ???? ?????????? ????????",
        "?????????????????????? ???????????????????? ?????????????????????? ???? ?????? ???? ???????? ??????????????????"
    ]
    current_tab = st.sidebar.radio('?????????? ??????????????', tabs)
    if current_tab == tabs[0]:
        st.write("???????? ?????????????? ???????????? ???? ?????????????? ???????????? ??????????????????, ?????????? ?????????? ???????????? ???????? ?????????????????????? ????????, ?????????? ?????????????? ???????????????? ?????????????????? ???????????? ?????????????????????? ?????????? ?????????????????? ?????????? ??????????. ????????????????: [Kaggle](https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016).")
        # here we construct all descriptions and display in a pretty way
        data_description = "\n".join(f"* **{key} - {value}**" for key, value in ru_columns.items())
        st.subheader('?????????????? ????????????')
        st.write(data[:30])
        st.subheader('???????????? ???????????? ????????????:')
        st.write(data_description)

        
            
    elif current_tab == tabs[1]:
        st.subheader('???????????????? ?????????????????? ??????????????????????')
        col1, col2, col3 = st.columns(3)
        with col1:
            # col_1 = st.columns(1)
            categories = ['sex', 'age', 'generation']
            feature = st.radio(
                "???????????????? ??????????????",
                [f"{key}: {value}" for key,value in ru_columns.items() if key in categories]).split(':')[0]
            
        with col2:
            linear = st.selectbox(
                    '???????????????? ?????? ?????????????? ?????? ????????????????????',
                    ('????????????????', '???????????????? ??????????????????')) == '????????????????'
        with col3:
            targets = {'???????????????????? ??????????????????????': "suicides_no", "???????????????????? ?????????????????????? ???? 100 ??????": "suicides/100k pop"}
            target_description = col3.radio("", list(targets.keys()))
            target = targets[target_description]
        fig_1 = plt.figure(figsize=(10, 4))
        if linear:
            unique_data = sorted(data[feature].unique())
            for category in unique_data:
                category_data = data[data[feature] == category]
                sns.lineplot(category_data.year, category_data[target], ci = None)
            plt.legend(unique_data)
        else:
            data_pie = data[feature].value_counts().to_frame()
            plt.pie(data_pie[feature].tolist(), labels = data_pie.index.tolist(), textprops={"fontsize":5})
        st.pyplot(fig_1)


    elif current_tab == tabs[2]:
        st.subheader('?????????????????????????? ???????????? ???????????????????? ?????????????????????? ???? ??????????????')
        start, end = st.slider("???????? ????????????????????", 1985, 2016, (1985, 2016))
        col1, col2 = st.columns(2)
        targets = {'???????????????????? ??????????????????????': "suicides_no", "???????????????????? ?????????????????????? ???? 100 ??????": "suicides/100k pop"}
        target_description = col1.radio("?????????????? ????????????????????", list(targets.keys()))
        countries = col2.multiselect("????????????", sorted(data['country'].unique()), help="???????????????? ???????????? ?????? ??????????????????")
        target = targets[target_description]
        fig = plt.figure(figsize=(10, 4))
        for country in countries:
            country_data = data[(data['country'] == country) & (start <= data['year']) & (data['year'] <= end)]
            sns.lineplot(country_data.year, country_data[target], ci = None)
        plt.legend(countries)
        st.pyplot(fig)
        
    elif current_tab == tabs[3]:
        st.subheader('?????? ?????????? ???? ???????????????????? ??????????????????????')
        year = st.slider("???????????????? ??????", 1985, 2015, 2015)
        
        ### aggregated data
        year_data = data[data['year'] == year]
        suicide_number = year_data.groupby(["country","year"])["suicides_no"].sum()
        suicide_number_sum_aggregated = suicide_number.sort_index(ascending=True)[:] * 100
        population = year_data.groupby(["country","year"]).population.sum().sort_index(ascending=False)[:]
        df_total = suicide_number_sum_aggregated / population

        country_dict={}
        for country in df_total.index.get_level_values(0):
            if country not in country_dict.keys():
                country_dict[country] = df_total[country].mean()

        tup = sorted(list(country_dict.items()), key= lambda pair:pair[1], reverse = True)
        

        country_list = [a[0] for a in tup]
        country_suicide = [a[1] for a in tup]
        fig = plt.figure(figsize=(8,32))
        new_df = pd.DataFrame(tup, columns=['country', 'n_suicides'])


        chart = (
            alt.Chart(
                new_df,
            )
            .mark_bar()
            .encode(
                x=alt.X('n_suicides', title="#"),
                y=alt.Y(
                    "country",
                    sort=alt.EncodingSortField(field='n_suicides', order="descending"),
                    title="",
                ),
                color=alt.Color(
                    'n_suicides',
                    legend=alt.Legend(title="#"),
                ),
                tooltip=["country", 'n_suicides'],
            )
        )

        st.altair_chart(chart, use_container_width=True)

    elif current_tab == tabs[4]:
        st.subheader('?????????????????????? ???????????? ???? ?????????? ????????')
        aggregated_data = make_aggregated_data(data)
        countries = {}
        for country in pycountry.countries:
            countries[country.name] = country.alpha_3
        country_dict = dict()
        for index, row in aggregated_data.iterrows():
            try:
                year = row['year']
                name = row['country']
                rate = row['suicides_by_100k']
                iso_code = countries[name]
                if iso_code not in country_dict:
                    country_dict[iso_code] = (year, rate, name)
                else:
                    old_year, _, _ = country_dict[iso_code]
                    if year > old_year:
                        country_dict[iso_code] = (year, rate, name)
            except KeyError:
                continue # in case we have incorrect caption

        country_map = {}
        country_map["iso_a3"] = list(country_dict.keys())
        country_map["suicide_rate"] = list(x[1] for x in country_dict.values())
        country_map["year"] = list(x[0] for x in country_dict.values())
        country_map["name"] = list(x[2] for x in country_dict.values())
        country_map_df = pd.DataFrame(country_map)

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))
        
        result = pd.merge(world, country_map_df, on = "iso_a3")
        fig = px.choropleth(result, locations='iso_a3', color='suicide_rate',
                            color_continuous_scale="ylorbr",
                            range_color=(0, 0.04),
                            labels={'iso_a3':'?????? ISO', 'name_y': '????????????????', 'suicide_rate': '??????????????????????'}
                            )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)


    elif current_tab == tabs[5]:
        st.subheader('?????????????????????? ???????????????????? ?????????????????????? ???? ?????? ???? ???????? ??????????????????')
        aggregated_data = make_aggregated_data(data)
        col1, col2 = st.columns(2)
        country = col1.selectbox("???????????????? ????????????", sorted(aggregated_data['country'].unique()), help="")
        targets = {'???????????????????? ??????????????????????': "n_suicides", "???????????????????? ?????????????????????? ???? 100 ??????": "suicides_by_100k"}
        target_description = col2.radio("????????", list(targets.keys()))
        target = targets[target_description]
        
        # ????????????????: https://plotly.com/python/multiple-axes/
        country_data = aggregated_data[aggregated_data['country'] == country]
        ### FROM: https://plotly.com/python/multiple-axes/
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=country_data['year'], y=country_data[target]),
            secondary_y=False, 
        )
        fig.add_trace(
            go.Scatter(x=country_data['year'], y=country_data['gdp_per_capita']),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="??????")
        fig.update_yaxes(title_text=f"<b>{target_description}</b>", secondary_y=False, color='#636EFA')
        fig.update_yaxes(title_text="<b>??????</b> ???? ???????? ??????????????????", secondary_y=True, color='#EF553B')
        ### ENDFROM
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)







