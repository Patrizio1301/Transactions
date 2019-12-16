import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.core.display import display, HTML
import plotly.figure_factory as ff

df = [
    dict(Task='III Deployment', Start='2020-03-01', Finish='2020-03-31', Resource='PhaseThree'),
    dict(Task='III Deployment', Start='2020-03-01', Finish='2020-03-31', Resource='PhaseThree'),
    dict(Task='III Deployment', Start='2020-03-01', Finish='2020-03-31', Resource='PhaseThree'),
    dict(Task='III Deployment', Start='2020-03-01', Finish='2020-03-31', Resource='PhaseThree'),
    dict(Task='II Modeling', Start='2020-01-31', Finish='2020-02-28', Resource='PhaseTwo'),
    dict(Task='II Modeling', Start='2020-01-31', Finish='2020-02-28', Resource='PhaseTwo'),
    dict(Task='II Modeling', Start='2020-01-31', Finish='2020-02-28', Resource='PhaseTwo'),
    dict(Task='I Preprocessing', Start='2020-01-20', Finish='2020-01-31', Resource='PhaseOne', text="uhdaushdakd", hoverinfo="text+x+y"),
    dict(Task='I Preprocessing', Start='2020-01-10', Finish='2020-01-25', Resource='PhaseOne'),
    dict(Task='I Preprocessing', Start='2020-01-01', Finish='2020-01-15', Resource='PhaseOne')
]

colors = dict(PhaseOne='rgb(46, 137, 205)',
              PhaseTwo='rgb(114, 44, 121)',
              PhaseThree='rgb(198, 47, 105)')

annots =  [
        dict(x='2020-03-16', align="center", y=0,text="Task label A", showarrow=False, font=dict(color='white')),
        ]

# plot figure


fig = ff.create_gantt(df, colors=colors, index_col='Resource', title='Action Plan',
                      show_colorbar=True, bar_width=0.3, showgrid_x=True, showgrid_y=True)

fig['layout']['annotations'] = annots
fig.update_layout(
    margin=dict(l=20, r=20, t=80, b=0),
    paper_bgcolor="white",
)
fig.show()