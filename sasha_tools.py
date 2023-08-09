import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss


__all__ = ['split_group', 
           'get_bootstrap_indices',
           'one_dim_analysis']

def split_group(df,
                prop,
                weight,
                quantiles = True,
                weighted_quantiles = False,
                n_groups = 5,
                values = None
                ):
    #Creates an additional binarized feature prop + '_group'
    #Supports quantiles, weighted quantile and values binarization (TDDO)
    df.sort_values(prop, inplace = True)
    if quantiles:
        if df[prop].nunique() <= n_groups:
            values = sorted(df[prop].unique())
        else:
            values = [df[prop].quantile((i + 1) / n_groups) for i in range(n_groups)]

    values = [df[prop].min() - 1] + values
    for i in range(len(values) - 1):
        df.loc[(df[prop] > values[i]) & (df[prop] <= values[i + 1]), prop + '_group'] =\
              round(df.loc[(df[prop] > values[i]) & (df[prop] <= values[i + 1]), prop].mean(), 3)

    df.loc[df[prop] > values[i + 1], prop + '_group'] = round(df.loc[df[prop] > values[i + 1], prop].mean(), 3)
        
    return df


def get_bootstrap_indices(indices, size = 1000):
    return np.array([np.random.choice(indices, replace=True, size = len(indices)) for _ in range(size)])






def one_dim_analysis(df,
                     prop,
                     weight,
                     target,
                     time,
                     split_group_flag = True,
                     n_groups = 5,
                     time_flag = True,
                     bootstrap = False
                     ):
    if split_group_flag:
        df = split_group(df,
                         prop = prop,
                         weight = weight,
                         quantiles = True,
                         n_groups = n_groups
                         )
    else:
        df[prop + '_group'] = df[prop]
        
    df_plot1 = df.groupby(prop + '_group')[[target, weight]].sum()
    df_plot1[target + '_freq'] = df_plot1[target] / df_plot1[weight]
    df_plot1.sort_index(inplace = True)
    #df_plot1.reset_index(inplace = True)
    #df_plot1.sort_values(by = prop + '_group', inplace = True)


    if bootstrap:
        for val in df_plot1.index: #df_plot1[prop + '_group'].values:
            bootstrap_indx = df[df[prop + '_group'] == val].index
            bootstrap_indecies = get_bootstrap_indices(bootstrap_indx, 200)
            bootstrap_freqs = []
            for indx in bootstrap_indecies:
                bootstrap_freqs.append(df.loc[indx, target].sum() / df.loc[indx, weight].sum())
            df_plot1.loc[val, target + '_freq_q025'] = np.quantile(bootstrap_freqs, q = 0.025)
            df_plot1.loc[val, target + '_freq_q975'] = np.quantile(bootstrap_freqs, q = 0.975)

    else:
        for val in df_plot1.index:
            df_plot1.loc[val, target + '_freq_q025'] = df_plot1.loc[val, target + '_freq'] -\
                                                         1.96 * np.sqrt(df_plot1.loc[val, target + '_freq'] * (1 - df_plot1.loc[val, target + '_freq']) /\
                                                                         df_plot1.loc[val, weight])
            
            df_plot1.loc[val, target + '_freq_q975'] = df_plot1.loc[val, target + '_freq'] +\
                                                         1.96 * np.sqrt(df_plot1.loc[val, target + '_freq'] * (1 - df_plot1.loc[val, target + '_freq']) /\
                                                                         df_plot1.loc[val, weight])
    
    
    fig, axes = plt.subplots(1, 2, figsize = (18, 9))
    
    
    if df_plot1.index.dtype == 'float64' or df_plot1.index.dtype == 'int64':
        x_values = df_plot1.index
    else:
        x_values = list(range(len(df_plot1)))

    ax_temp = axes[0].twinx() 
    ax_temp.set_ylabel('Quantity')

    width = 0.3 *(x_values[-1] -x_values[0]) / n_groups
    for i in range(len(df_plot1)):
        ax_temp.bar(x_values[i], df_plot1.iloc[i][weight],color = 'green', alpha = 0.15, width = width)

    for i in range(len(df_plot1) - 1):
        axes[0].plot([x_values[i], x_values[i + 1]], 
                 [np.log(df_plot1.iloc[i][target + '_freq'] / (1 - df_plot1.iloc[i][target + '_freq'])),
                   np.log(df_plot1.iloc[i + 1][target + '_freq'] / (1 - df_plot1.iloc[i + 1][target + '_freq']))],
                 color = 'blue', alpha = 0.7
                 )


    for i in range(len(df_plot1)):
        axes[0].plot([x_values[i], x_values[i]],
                 [np.log(df_plot1.iloc[i][target + '_freq_q025'] / (1 - df_plot1.iloc[i][target + '_freq_q025'])),
                   np.log(df_plot1.iloc[i][target + '_freq_q975'] / (1 - df_plot1.iloc[i][target + '_freq_q975']))],
                 color = 'blue', alpha = 0.5
                 )
    
    axes[0].set_ylabel('Logit(' + target + '_freq)')
    ax_temp.set_ylabel('Qunatity')
    axes[0].set_xlabel(prop + '_group')
    axes[0].set_xticks(x_values, df_plot1.index, rotation = 60)
    axes[0].set_title(target + '_freq' + ' by ' + prop)

    if time_flag:
        df_plot2 = df.groupby([time, prop + '_group'])[[target, weight]].sum()
        df_plot2[target + '_freq'] = df_plot2[target] / df_plot2[weight]
        df_plot2.sort_index(inplace = True)

        #df_plot2.reset_index(inplace = True)
        #df_plot2.sort_values([time, prop + '_group'])

        for t, v in df_plot2.index:
            df_plot2.loc[(t, v), target + '_freq_q025'] = df_plot2.loc[(t, v), target + '_freq'] -\
                                                             1.96 * np.sqrt(df_plot2.loc[(t, v), target + '_freq'] * (1 - df_plot2.loc[(t, v), target + '_freq']) /\
                                                                            df_plot2.loc[(t, v), weight]
                                                                            )
            df_plot2.loc[(t, v), target + '_freq_q975'] = df_plot2.loc[(t, v), target + '_freq'] +\
                                                             1.96 * np.sqrt(df_plot2.loc[(t, v), target + '_freq'] * (1 - df_plot2.loc[(t, v), target + '_freq']) /\
                                                                            df_plot2.loc[(t, v), weight]
                                                                            )
            
        
        colors = ['dimgrey', 'chocolate', 'palegreen', 'lightseagreen', 'cyan', 'navy', 'plum'] + ['black'] * 5
        df_time_weight = df_plot2.reset_index().groupby(time)[weight].sum()
        x_values = sorted(df[time].unique())
        y_values = sorted(df[prop + '_group'].unique())
        ax_temp = axes[1].twinx()
        for i, _ in enumerate(x_values[:-1]):
            cum_weight_1, cum_weight_2 = 0, 0
            for j, y in enumerate(y_values):
                axes[1].plot([i, i + 1], 
                           [df_plot2.loc[(x_values[i], y_values[j]), target + '_freq'], df_plot2.loc[(x_values[i + 1], y_values[j]), target + '_freq']],
                           color = colors[j],
                           alpha = 0.8
                           )
                ax_temp.fill_between([i, i + 1],
                                    [cum_weight_1, cum_weight_2],
                                    [cum_weight_1 + df_plot2.loc[(x_values[i], y_values[j]), weight] / df_time_weight.loc[x_values[i]],
                                     cum_weight_2 + df_plot2.loc[(x_values[i + 1], y_values[j]), weight] / df_time_weight.loc[x_values[i + 1]]
                                     ],
                                     color = colors[j],
                                     alpha = 0.15
                                    )
                cum_weight_1 += df_plot2.loc[(x_values[i], y_values[j]), weight] / df_time_weight.loc[x_values[i]]
                cum_weight_2 += df_plot2.loc[(x_values[i + 1], y_values[j]), weight] / df_time_weight.loc[x_values[i + 1]]
        
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                if i == 0:
                    axes[1].plot([i, i],
                              [df_plot2.loc[(x, y), target + '_freq_q025'], df_plot2.loc[(x, y), target + '_freq_q975']],
                              color = colors[j],
                              alpha = 0.8,
                              label = y
                            )
                else:
                    axes[1].plot([i, i],
                              [df_plot2.loc[(x, y), target + '_freq_q025'], df_plot2.loc[(x, y), target + '_freq_q975']],
                              color = colors[j],
                              alpha = 0.8
                            )
        
        axes[1].set_title('Time stability of ' + target + '_freq by' + prop + '_group')
        axes[1].set_xticks(range(len(x_values)), x_values)
        axes[1].set_ylabel(target + '_freq')
        ax_temp.set_ylabel('Population fraction')
        axes[1].legend()
            

        


    fig.tight_layout()
    plt.show()
    return df_plot1, df_plot2
        
    

