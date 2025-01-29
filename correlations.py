def find_correlations(df):
    correlation_matrix = df.corr().round(3).to_html()
    print(correlation_matrix)