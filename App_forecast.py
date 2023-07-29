#%%
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import scipy as sp
import plotly.graph_objects as go
from pathlib import Path

#%%
image = Image.open(Path(__file__).parent / 'Logo_blanco.jpeg')
st.image(image, width=150)
st.title("Modelo Suprabayesiano para la estimación del crecimiento de Arequipa")
st.header("1. Etapa de calibración")
st.markdown("Con el objetivo de calibrar sus opiniones y sumarlas al modelo, coloque a continuación \
        las tasas de crecimiento que UD. PIENSA que tuvo cada uno de los siguientes sectores en Arequipa \
        en 2022. Coloque solo números, sin el símbolo '%' y en puntos porcentuales. Ejemplo: En lugar de -2.5%, solo coloque -2.5")
df_gsectorial = pd.DataFrame({2022:[-0.16, 8.63, 1.51, 7.31, 3.05, 3.66]}, index=['Agropecuaria','Minería','Manufactura','Construcción','Comercio','Otros servicios'])
df_e1 = pd.DataFrame({'Growth_expected':[-0.2,8.,1.5,7.,2.,3.]}, index=df_gsectorial.index)
df_growth = pd.DataFrame.from_dict(dict([[2004,5.449063],[2005,6.842634],[2006,6.101323],[2007,15.599755],[2008,11.146391],
                          [2009,0.776626],[2010,5.917537],[2011,4.365751],[2012,4.728066],[2013,2.702974],
                          [2014,0.637255],[2015,3.298967],[2016,25.924020],[2017,3.719005],[2018,2.545244],
                          [2019,-0.325247],[2020,-15.644018],[2021,13.206235],[2022,4.963000]]), orient='index', columns=['Crecimiento'])

df_e2 = pd.DataFrame(index=df_gsectorial.index, columns=['Tasa de crecimiento en 2022 (puntos porcentuales)'])
edited_df_e2 = st.data_editor(df_e2)
edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'] = edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'].astype(float)
if edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'].hasnans:
    st.markdown("Escriba valores en todas las celdas")
else:
    try:
        # edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'] = edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'].astype(float)
        var_e1 = ((df_e1['Growth_expected'] - df_gsectorial[2022])**2).sum()
        var_e2 = ((edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'].astype(float) - df_gsectorial[2022])**2).sum()
        cov_e1e2 = ((df_e1['Growth_expected'] - df_gsectorial[2022])*(edited_df_e2['Tasa de crecimiento en 2022 (puntos porcentuales)'].astype(float) - df_gsectorial[2022])).sum()
        Sigma_e1e2 = np.array([[var_e1, cov_e1e2],[cov_e1e2, var_e2]])

        sigma_inv = np.linalg.inv(Sigma_e1e2)
        mu_0 = -3.489 # marzo 2023
        sigma_0 = 2.025# 4.1 # marzo 2023

        xe1 = np.outer(np.linspace(mu_0-3*np.sqrt(var_e1), mu_0+3*np.sqrt(var_e1), 30), np.ones(30))
        xe2 = np.outer(np.linspace(mu_0-3*np.sqrt(var_e2), mu_0+3*np.sqrt(var_e2), 30), np.ones(30)).T
        flikelihood = sp.stats.multivariate_normal(mean=[mu_0, mu_0], cov=Sigma_e1e2).pdf(np.dstack((xe1, xe2)))
        zmax = flikelihood.max()
        graph_likelihood = go.Figure(data=[go.Surface(x=xe1, y=xe2, z=flikelihood, colorscale='Electric')])
        graph_likelihood.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
        graph_likelihood.update_layout( 
        scene=dict(
            xaxis_title='Opinión del experto 1',
            yaxis_title='Opinión suya',
            zaxis_title='Densidad (Likelihood)',
        ),
        scene_camera_eye=dict(x=0, y=0, z=100*zmax)
        )
        st.subheader("1.1. Gráfico de la distribución de sus opiniones y la del experto_1 (Likelihood)")
        st.markdown("Su opinión se calibró con las opiniones de otros analistas mediante el enfoque Suprabayesiano. \
                Por simplicidad, solo se considerará solo un analista adicional.")
        st.plotly_chart(graph_likelihood)
        
    except:
        st.markdown("escriba solo números")

st.header("2. Etapa de agregación de opiniones al modelo")
forecast_e2_str = st.text_input("¿Cuál considera que será el crecimiento de la economía de Arequipa en 2023? Coloque el valor sin el símbolo % (Ejemplo, en lugar de colocar -2.5%, coloque -2.5)", '')
st.header("3. Resultados")
if forecast_e2_str!='':
    forecast_e1 = -1.5
    forecast_e2 = float(forecast_e2_str)
    array_expert = np.array([forecast_e1, forecast_e2])############################
    
    mu_new = ((array_expert.T@sigma_inv).sum() + (sigma_0**(-2))*mu_0)/(sigma_inv.sum() + sigma_0**(-2))
    sigma_new = (sigma_inv.sum() + sigma_0**(-2))**(-1)

    # prior = np.random.normal(loc=mu_0, scale=sigma_0, size=1000)
    # likelihood = np.random.multivariate_normal(mean=[mu_0, mu_0], cov=Sigma_e1e2, size=1000)
    # posterior = np.random.normal(loc=mu_new, scale=scale_new, size=1000)

    x_prior = np.linspace(mu_0-3*sigma_0, mu_0+3*sigma_0, 100)
    x_posterior = np.linspace(mu_new-3*sigma_new, mu_new+3*sigma_new, 100)
    fprior = sp.stats.norm(loc=mu_0, scale=sigma_0).pdf(x_prior)
    fposterior = sp.stats.norm(loc=mu_new, scale=sigma_new).pdf(x_posterior)

    
    st.subheader("3.1. Gráfico de la distribución inicial (a priori) y la distribución final considerando la data y las opiniones de los expertos (a posteriori)")
    st.markdown("El pronóstico actualizado utiliza la información de la etapa de calibración para darle un peso mayor al \
            experto que obtuvo mejores pronósticos en la calibración y considera la correlación entre ambas opiniones (que puede ser negativa o positiva)")
    # fig = plt.figure(figsize=(10, 4))
    # sns.kdeplot(prior, fill=True)
    # sns.kdeplot(posterior, fill=True)
    # plt.plot(x_prior, fprior, alpha=0.5)
    # plt.fill_between(x_prior, fprior, alpha=0.5)
    # plt.plot(x_posterior, fposterior, alpha=0.5)
    # plt.fill_between(x_posterior, fposterior, alpha=0.5)
    # plt.axvline(x=mu_0, ymin=0, ymax=fprior.max(), linestyle='dashed', color='gray')
    # plt.axvline(x=forecast_e1, ymin=0, ymax=fprior.max(), linestyle='dashed', color='gray')
    # plt.axvline(x=forecast_e2, ymin=0, ymax=1, linestyle='dashed', color='gray')
    # plt.axvline(x=mu_new, ymin=0, ymax=fposterior.max(), linestyle='dashed', color='gray')
    # plt.xlim(-20, 20)
    # st.pyplot(fig)
    graph_line = go.Figure()
    # probs_prior_acum = dict(zip(fprior, fprior.cumsum()/fprior.sum()))
    probs_prior_acum = fprior.cumsum()/fprior.sum()*100
    probs_posterior_acum = fposterior.cumsum()/fposterior.sum()*100

    graph_line.add_trace(go.Scatter(x=x_prior, y=fprior, fill='tozeroy', name='Basado en datos',
                                    fillcolor='rgba(0, 0, 255, 0.2)', line_color='rgba(0, 0, 255, 0.5)',
                                    text=probs_prior_acum))
    graph_line.add_trace(go.Scatter(x=x_posterior, y=fposterior, fill='tozeroy', name='Basado en datos y opiniones de expertos',
                                    fillcolor='rgba(255, 0, 0, 0.2)', line_color='rgba(255, 0, 0, 0.5)',
                                    text=probs_posterior_acum))
    graph_line.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255, 255, 255, 0.1)'
        ))
    # graph_line.add_vline(x=mu_0, line_width=2, line_dash="dash", line_color="rgba(0, 0, 255, 1)",
    #                      annotation=dict(text=f"Pronóstico inicial: {round(mu_0,1)}", font=dict(color="blue")), annotation_position="bottom left"
    #                      )
    graph_line.add_vline(x=forecast_e1, line_width=1, line_dash="dash", line_color="grey",
                         annotation=dict(text=f"Experto 1: {forecast_e1}", font=dict(color="grey")), annotation_position="top left", 
                         )
    graph_line.add_vline(x=forecast_e2, line_width=1, line_dash="dash", line_color="green",
                         annotation=dict(text=f"Usted: {forecast_e2}", font=dict(color="green")), annotation_position="top left"
                         )
    # graph_line.add_vline(x=mu_new, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 1)",
    #                      annotation=dict(text=f"Pronóstico final: {round(mu_new,1)}", font=dict(color="red")), annotation_position="bottom left"
    #                      )
    graph_line.add_trace(go.Scatter(x=[mu_0], y=[0], mode = 'markers',
                         marker_size = 15, marker_color='blue', name='Pronóstico inicial', text=[50.]))
    graph_line.add_trace(go.Scatter(x=[mu_new], y=[0], mode = 'markers',
                         marker_size = 15, marker_color='red', name='Pronóstico final', text=[50.]))
    graph_line.update_traces(hovertemplate="Probabilidad de crecimiento menor a %{x:.2f}: %{text:.2f}%")
    graph_line.update_layout(hovermode="x unified")
    graph_line.update_layout(
        xaxis_title="Tasa de crecimiento pronosticada en 2023",
        yaxis_title="Densidad")
    st.plotly_chart(graph_line)
    # st.markdown(f"mu_0={mu_0}; mu_new={mu_new}; e1={forecast_e1}; e2={forecast_e2}")
    st.subheader("3.2. Gráfico del pronóstico del crecimiento de la economía de Arequipa")
    st.markdown(f"Considerando ambas opiniones, el **nuevo pronóstico es de {round(mu_new,1)}%** para el crecimiento de \
            la economía de Arequipa en 2023. El **pronóstico inicial fue de {round(mu_0,1)}%.**")
    g2022 = df_growth.loc[2022,'Crecimiento']
    graph_forecast = go.Figure()
    graph_forecast.add_trace(go.Scatter(x=df_growth.index, y=df_growth['Crecimiento'], name='Histórico', line_color='rgba(100, 100, 100, 0.5)', mode='lines+markers'))
    graph_forecast.add_trace(go.Scatter(x=[2022,2023, 2023, 2022], y=[g2022, mu_0+1.96*sigma_0, mu_0-1.96*sigma_0, g2022], fill='toself',
                                        fillcolor='rgba(0,0,255,0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        hoverinfo="skip",
                                        showlegend=False))
    graph_forecast.add_trace(go.Scatter(x=[2022, 2023, 2023, 2022], y=[g2022, mu_new+1.96*sigma_new, mu_new-1.96*sigma_new, g2022], fill='toself',
                                        fillcolor='rgba(255,0,0,0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        hoverinfo="skip",
                                        showlegend=False))
    graph_forecast.add_trace(go.Scatter(x=[2022,2023], y=[g2022, mu_0], name='Basado en datos', line_color='rgba(0, 0, 255, 0.5)', mode='lines+markers'))
    graph_forecast.add_trace(go.Scatter(x=[2022,2023], y=[g2022, mu_new], name='Basado en datos y opiniones', line_color='rgba(255, 0, 0, 0.5)', mode='lines+markers'))
    graph_forecast.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255, 255, 255, 0.1)'
        ))
    graph_forecast.update_layout(
        xaxis_title="Año",
        yaxis_title="Tasa de crecimiento (%)")
    st.plotly_chart(graph_forecast)
else:
    st.markdown("Escriba solo valores numéricos sin el símbolo % y en puntos porcentuales (Ejemplo: en lugar de 0.02 o 2%, escriba 2)")
