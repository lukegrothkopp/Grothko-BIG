from src.data_loader import load_dataframe

def test_load():
    df = load_dataframe('data/sample_sales.csv')
    assert len(df) > 0
    assert set(['date','region','product','customer_id','age','gender','quantity','sales','discount_pct']).issubset(df.columns)
