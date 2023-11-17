import streamlit as st
import pandas as pd
from pyathena import connect

# AWS Athena connection parameters

aws_access_key_id = 'AKIA5WIK7ZBXV4OGFNFF'
aws_secret_access_key = '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'
region_name = 'ap-south-1'
database = 'stock_market'
table = 'kafka_stock_market_using_aws_ms_csv'

# Establish connection to Athena
conn = connect(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
    s3_staging_dir='s3://athena-stock-market-kafka-ms/',
    schema_name=database,
)

st.title('Amazon Athena')

# Athena query input
choice=st.selectbox('Select Your Choice',['Inbuilt Commands','Enter Command'])
if choice == 'Inbuilt Commands':
    choose_query=st.selectbox('Select Query',['Show','Count','Index Name','Average','Min Close','Max Index','Max of each'])

    if choose_query == 'Show':
        query = 'SELECT * FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" limit 50;'
    elif choose_query == 'Count':
        query = 'SELECT count(*) FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" limit 10;'
    elif choose_query =='Index Name':
        query = 'SELECT DISTINCT "index" FROM  "stock_market"."kafka_stock_market_using_aws_ms_csv";'
    elif choose_query == 'Average':
        query ='SELECT "index", AVG("open") AS avg_open FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" GROUP BY "index" ORDER BY MAX("open") DESC;'
    elif choose_query == 'Min Close':
        query = 'SELECT "index", MIN("close") AS min_close FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" GROUP BY "index" ORDER BY MIN("close") DESC;'
    elif choose_query == 'Max Index':
        query = 'SELECT(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "open" = (SELECT MAX("open") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_open_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "high" = (SELECT MAX("high") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_high_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "low" = (SELECT MAX("low") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_low_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "close" = (SELECT MAX("close") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_close_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "adj close" = (SELECT MAX("adj close") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_adj_close_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "volume" = (SELECT MAX("volume") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_volume_index,(SELECT index FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" WHERE "closeusd" = (SELECT MAX("closeusd") FROM "stock_market"."kafka_stock_market_using_aws_ms_csv")) AS max_closeusd_index;'
    elif choose_query == 'Max of each':
        query = 'SELECT index,MAX("open") AS max_open,MAX("high") AS max_high,MAX("low") AS max_low,MAX("close") AS max_close,MAX("adj close") AS max_adj_close,MAX("volume") AS max_volume,MAX("closeusd") AS max_closeusd FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" Group by index;'

elif choice == 'Enter Command':
    query = st.text_input('Enter your Athena SQL query:', 'SELECT * FROM "stock_market"."kafka_stock_market_using_aws_ms_csv" LIMIT 10')



# Execute the query and fetch the results
try:
    with conn.cursor() as cursor:
        cursor.execute(query)

        # Convert results to a DataFrame
        results_df = pd.read_sql_query(query, conn)

    # Display the results in Streamlit
    st.dataframe(results_df)

except Exception as e:
    st.error(f"Error executing query: {e}")
