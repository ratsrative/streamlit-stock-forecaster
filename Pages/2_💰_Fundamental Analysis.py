import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from st_aggrid import AgGrid


class InvalidCompanyNameException(Exception):
    pass

@st.cache_data
def get_nifty500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500", match='Nifty 500 List')
    df = df[0]
    df.columns = df.iloc[0]
    df["Symbol"] = df["Symbol"] + '.NS'
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Company Name"])
    )
    return tickers, tickers_companies_dict


@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)


image1 = Image.open('./Pages/Stock Market Analysis Header.png')
st.image(image1)
st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_nifty500_components()

tickers = st.sidebar.selectbox(
    "Ticker",
    available_tickers,
    format_func=tickers_companies_dict.get
)
ticker = yf.Ticker(tickers)

## inputs for technical analysis

st.sidebar.header("Quantitative Financial Reports")
financials = st.sidebar.selectbox("Add Financials", ["Income Statement",
                                                     "Quarterly Income Statement",
                                                     "Balance Sheet",
                                                     "Quarterly Balance Sheet",
                                                     "Cash Flow Statement",
                                                     "Quarterly Cash Flow Statement"])
# df = load_data(ticker, start_date, end_date)
title_str = st.header(f"{tickers_companies_dict[tickers]}'s {financials}")

st.write("""
### User manual
* you can select any of the companies that is a component of the **:green[NIFTY 500]** index
* you can download the selected Quantitative Financial documents as csv file for **Fundamental Analysis**
""")
fin_df = None
try:
    if tickers_companies_dict[tickers] == 'Company Name':
        raise InvalidCompanyNameException("Invalid company name: 'Company Name'")
    if financials == 'Income Statement':
        fin_df = ticker.income_stmt.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Income/Expenditure'}, inplace=True)
        AgGrid(fin_df,key="income_statement")

    elif financials == 'Quarterly Income Statement':
        fin_df = ticker.quarterly_income_stmt.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Income/Expenditure'}, inplace=True)
        AgGrid(fin_df,key="quarterly_income_statement")
    elif financials == 'Balance Sheet':
        fin_df = ticker.balance_sheet.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Assets & Liabilities & Equities'}, inplace=True)
        AgGrid(fin_df,key="balance_sheet")
    elif financials == 'Quarterly Balance Sheet':
        fin_df = ticker.quarterly_balance_sheet.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Assets & Liabilities & Equities'}, inplace=True)
        AgGrid(fin_df,key="quarterly_balance_sheet")
    elif financials == 'Cash Flow Statement':
        fin_df = ticker.cash_flow.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Cash Inflow & Outflow'}, inplace=True)
        AgGrid(fin_df,key="cash_flow_statement")
    elif financials == 'Quarterly Cash Flow Statement':
        fin_df = ticker.quarterly_cashflow.reset_index()
        fin_df.columns = fin_df.columns.astype(str)
        fin_df.rename(columns={'index': 'Cash Inflow & Outflow'}, inplace=True)
        AgGrid(fin_df,key="quarterly_cash_flow_statement")
except(AttributeError, KeyError, IndexError) as e:
    st.write(f"Error: {e}")
    st.write("Unable to fetch financial data for the selected company.")

if fin_df is not None:
    @st.cache_data
    def convert_df(dtf):
        return dtf.to_csv(index=False).encode('utf-8')


    if fin_df is not None:
        csv = convert_df(fin_df)
        st.download_button(
            label="Download data as csv",
            data=csv,
            file_name=f"{financials}" + ".csv",
            mime='csv', )

# Download stock information
data_exp = st.expander("Preview Financial Ratios")
available_cols = ["Profitability Ratios",
                  "Leverage Ratios",
                  "Valuation Ratios",
                  "Operating Ratios",
                  "Piotroski Score",
                  "Intrinsic Value"]
columns_to_show = data_exp.multiselect(
    "Columns",
    available_cols,
    default='Profitability Ratios'
)

######################################### FUNDAMENTAL RATIOS ###########################################################

########################################### I. PROFITABILITY RATIOS#####################################################

## 1. PAT and PAT Growth (CAGR)
income_statement = ticker.income_stmt
income_statement = income_statement.reset_index()
# Calculate PAT (Profit After Tax)
pat_finalyr = float(income_statement[income_statement['index'] == 'Net Income'].iloc[:, 1])
# Calculate Revenue|
revenue = float(income_statement[income_statement['index'] == 'Total Revenue'].iloc[:, 1])
# Calculate PAT Margin (PAT / Revenue)
pat_margin = round((pat_finalyr / revenue) * 100, 2)  # Multiply by 100 to express in percentage
pat_3yrs_then = float(income_statement[income_statement['index'] == 'Net Income'].iloc[:, 3])

# Calculate CAGR (Compound Annual Growth Rate) for PAT
# Assuming annual data is available, calculate the number of periods (years)
num_periods = 3

# Calculate CAGR using the formula (end value / start value)^(1/number of periods) - 1
cagr_pat = round(((pat_finalyr / pat_3yrs_then) ** (1 / num_periods)) - 1, 2) * 100

## 2.EBITDA Margin (Operating Profit Margin) And EBITDA Growth (CAGR)

financials = ticker.financials
financials = financials.reset_index()
EBITDA_final_yr = float(financials[financials['index'] == 'EBITDA'].iloc[:, 1])
Operating_Revenue = float(financials[financials['index'] == 'Operating Revenue'].iloc[:, 1])
EBITDA_Margin = round((EBITDA_final_yr / Operating_Revenue) * 100, 2)
EBITDA_3yrs_then = float(financials[financials['index'] == 'EBITDA'].iloc[:, 3])
cagr_EBITDA = ((EBITDA_final_yr / EBITDA_3yrs_then) ** (1 / num_periods)) - 1
cagr_EBITDA = round(cagr_EBITDA * 100, 2)
# Print the results
# print("PAT Margin:",EBITDA_Margin, "%")
# print("CAGR of EBITDA (for the past 3 years):", cagr_EBITDA * 100, "%")

## 3.Return On Equity (ROE)

net_income_final = float(income_statement[income_statement['index'] == 'Net Income'].iloc[:, 1])
balance_sheet = ticker.balance_sheet.reset_index()
Stockholders_Equity = float(balance_sheet[balance_sheet['index'] == 'Stockholders Equity'].iloc[:, 1])
ROE = round((net_income_final / Stockholders_Equity) * 100, 2)

## 4.Return On Assets (ROA)

net_income = float(financials[financials['index'] == 'Net Income'].iloc[:, 1])
balance_sheet = ticker.balance_sheet.reset_index()
Total_assets = float(balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 1])
ROA = round((net_income / Total_assets) * 100, 2)

## 5.Return On Capital Employed (ROCE)

# Extract necessary financial metric
EBIT_final_yr = float(financials[financials['index'] == 'EBIT'].iloc[:, 1])
balance_sheet = ticker.balance_sheet.reset_index()
Current_Liabilities = float(balance_sheet[balance_sheet['index'] == 'Current Liabilities'].iloc[:, 1])
Capital_Employed = Total_assets - Current_Liabilities
ROCE = round((EBIT_final_yr / Capital_Employed) * 100, 2)

##############################################II.LEVERAGE RATIOS########################################################
## 1.Interest Coverage ratio
interest_expense = float(income_statement[income_statement["index"] == "Interest Expense"].iloc[:, 1])
Interest_Coverage_ratio = round((EBIT_final_yr / interest_expense), 2)

## 2.Debt to Equity ratio
Stockholders_Equity = float(balance_sheet[balance_sheet["index"] == "Stockholders Equity"].iloc[:, 1])
Total_Debt = float(balance_sheet[balance_sheet["index"] == "Total Debt"].iloc[:, 1])

Capital_Lease_Obligations = float(balance_sheet[balance_sheet["index"] == "Capital Lease Obligations"].iloc[:, 1])

try:
    Long_Term_Debt = float(balance_sheet[balance_sheet["index"] == "Long Term Debt"].iloc[:, 1])
except TypeError:
    Long_Term_Debt = 0

try:
    Current_Debt_And_Capital_Lease_Obligation = float(
        balance_sheet[balance_sheet["index"] == "Current Liabilities"].iloc[:, 1])
except TypeError:
    Current_Debt_And_Capital_Lease_Obligation = 1
try:
    Current_Liabilities = float(balance_sheet[balance_sheet["index"] == "Current Liabilities"].iloc[:, 1])
except TypeError:
    Current_Liabilities = 0

try:
    Long_Term_Deferred_Taxes_Liabilities = float(
        balance_sheet[balance_sheet["index"] == "Long Term Deferred Taxes Liabilities"].iloc[:, 1])
except TypeError:
    Long_Term_Deferred_Taxes_Liabilities = 0

try:
    Dividends_Payable = float(balance_sheet[balance_sheet["index"] == "Dividends Payable"].iloc[:, 1])
except TypeError:
    Dividends_Payable = 0

try:
    Total_Tax_Payable = float(balance_sheet[balance_sheet["index"] == "Total Tax Payable"].iloc[:, 1])
except TypeError:
    Total_Tax_Payable = 0

try:
    Accounts_Payable = float(balance_sheet[balance_sheet["index"] == "Accounts Payable"].iloc[:, 1])
except TypeError:
    Accounts_Payable = 0

try:
    Total_Liabilities_Net_Minority_Interest = float(
        balance_sheet[balance_sheet["index"] == "Total Liabilities Net Minority Interest"].iloc[:, 1])
except TypeError:
    Total_Liabilities_Net_Minority_Interest = 0

try:
    Total_Non_Current_Liabilities_Net_Minority_Interest = float(
        balance_sheet[balance_sheet["index"] == "Total Non Current Liabilities Net Minority Interest"].iloc[:, 1])
except TypeError:
    Total_Non_Current_Liabilities_Net_Minority_Interest = 0

try:
    Long_Term_Provisions = float(balance_sheet[balance_sheet["index"] == "Long Term Provisions"].iloc[:, 1])
except TypeError:
    Long_Term_Provisions = 0

try:
    Current_Debt = float(balance_sheet[balance_sheet["index"] == "Current Debt"].iloc[:, 1])
except TypeError:
    Current_Debt = 0

try:
    Derivative_Product_Liabilities = float(
        balance_sheet[balance_sheet["index"] == "Derivative Product Liabilities"].iloc[:, 1])
except:
    Derivative_Product_Liabilities = 0

try:
    Non_Current_Pension_And_Other_Postretirement_Benefit_Plans = float(
        balance_sheet[balance_sheet["index"] == "Non Current Pension And Other Postretirement Benefit Plans"].iloc[:,
        1])
except TypeError:
    Non_Current_Pension_And_Other_Postretirement_Benefit_Plans = 0

try:
    Trade_and_Other_Payables_Non_Current = float(
        balance_sheet[balance_sheet["index"] == "Trade and Other Payables Non Current"].iloc[:, 1])
except TypeError:
    Trade_and_Other_Payables_Non_Current = 0

try:
    Non_Current_Deferred_Revenue = float(
        balance_sheet[balance_sheet["index"] == "Non Current Deferred Revenue"].iloc[:, 1])
except TypeError:
    Non_Current_Deferred_Revenue = 0

try:
    Long_Term_Debt_And_Capital_Lease_Obligation = float(
        balance_sheet[balance_sheet["index"] == "Long Term Debt And Capital Lease Obligation"].iloc[:, 1])
except TypeError:
    Long_Term_Debt_And_Capital_Lease_Obligation = 0

try:
    Current_Notes_Payable = float(balance_sheet[balance_sheet["index"] == "Current Notes Payable"].iloc[:, 1])
except TypeError:
    Current_Notes_Payable = 0

try:
    Payables_And_Accrued_Expenses = float(
        balance_sheet[balance_sheet["index"] == "Payables And Accrued Expenses"].iloc[:, 1])
except TypeError:
    Payables_And_Accrued_Expenses = 0

try:
    Current_Accrued_Expenses = float(balance_sheet[balance_sheet["index"] == "Current Accrued Expenses"].iloc[:, 1])
except TypeError:
    Current_Accrued_Expenses = 0

try:
    Payables = float(balance_sheet[balance_sheet["index"] == "Payables"].iloc[:, 1])
except TypeError:
    Payables = 0

Long_Term_Capital_Lease_Obligation = float(
    balance_sheet[balance_sheet["index"] == "Long Term Capital Lease Obligation"].iloc[:, 1])

try:
    Current_Capital_Lease_Obligation = float(
        balance_sheet[balance_sheet["index"] == "Current Capital Lease Obligation"].iloc[:, 1])
except TypeError:
    Current_Capital_Lease_Obligation = 0

Total_Liabilities = Total_Debt + Capital_Lease_Obligations + Long_Term_Debt + Current_Debt_And_Capital_Lease_Obligation + Current_Liabilities + \
                    Long_Term_Deferred_Taxes_Liabilities + Dividends_Payable + Total_Tax_Payable + Accounts_Payable + \
                    Total_Liabilities_Net_Minority_Interest + Total_Non_Current_Liabilities_Net_Minority_Interest + \
                    Long_Term_Provisions + Current_Debt + Derivative_Product_Liabilities + Non_Current_Pension_And_Other_Postretirement_Benefit_Plans + \
                    Trade_and_Other_Payables_Non_Current + Non_Current_Deferred_Revenue + Long_Term_Debt_And_Capital_Lease_Obligation + \
                    Current_Notes_Payable + Payables_And_Accrued_Expenses + Current_Accrued_Expenses + Payables + Long_Term_Capital_Lease_Obligation + \
                    Current_Capital_Lease_Obligation

Debt_to_Equity_ratio = round(Total_Liabilities / Stockholders_Equity, 2)

## 3. Debt to Asset ratio

Debt_to_Asset_ratio = round(Total_Liabilities / Total_assets, 2)

###########################################III VALUATION RATIOS#########################################################
# 1.Price to Earnings(P/E) Ratio

PE_Ratio = round(ticker.info['trailingPE'], 2)

# 2.Price to Book Value(P/BV) Ratio

P_BV_Ratio = round(ticker.info['priceToBook'], 2)

# 3.Price to Sales(P/S) Ratio

P_Sales_Ratio = round(ticker.info['priceToSalesTrailing12Months'], 2)

########################################### IV OPERATING RATIOS #########################################################
# 1. Fixed Assets Turnover Ratio

Fixed_Assets = balance_sheet[balance_sheet['index'] == 'Net PPE'].iloc[:, 1].iloc[0]
Operating_Revenue = financials[financials['index'] == 'Operating Revenue'].iloc[:, 1].iloc[0]
Fixed_Assets_Turnover_Ratio = round(Operating_Revenue / Fixed_Assets, 2)

# 2. Total Assets Turnover Ratio
Total_Assets_Turnover_Ratio = round(Operating_Revenue / Total_assets, 2)

# 3. Working Capital Turnover Ratio

# Calculate Current Assets
Cash_And_Cash_Equivalents = balance_sheet[balance_sheet['index'] == 'Cash And Cash Equivalents'].iloc[:, 1].iloc[0]
Other_Short_Term_Investments = balance_sheet[balance_sheet['index'] == 'Other Short Term Investments'].iloc[:, 1].iloc[
    0]
Accounts_Receivable = balance_sheet[balance_sheet['index'] == 'Accounts Receivable'].iloc[:, 1].iloc[0]
Other_Receivables = balance_sheet[balance_sheet['index'] == 'Other Receivables'].iloc[:, 1].iloc[0]
try:
    Receivables = balance_sheet[balance_sheet['index'] == 'Receivables'].iloc[:, 1].iloc[0]
except:
    Receivables = 0
try:
    Prepaid_Assets = balance_sheet[balance_sheet['index'] == 'Receivables'].iloc[:, 1].iloc[0]
except:
    Prepaid_Assets = 0

    Current_Assets = Cash_And_Cash_Equivalents + \
                     Other_Short_Term_Investments + Accounts_Receivable + Other_Receivables + Receivables + Prepaid_Assets

# Calculate Current Liabilities
if 'Current Debt' in balance_sheet['index'].unique().tolist():
    Current_Debt = balance_sheet[balance_sheet['index'] == 'Current Debt'].iloc[:, 1].iloc[0]
else:
    Current_Debt = balance_sheet[balance_sheet['index'] == 'Current Debt And Capital Lease Obligation'].iloc[:, 1].iloc[
                       0] + \
                   balance_sheet[balance_sheet['index'] == 'Current Capital Lease Obligation'].iloc[:, 1].iloc[0]

try:
    Other_Current_Borrowings = balance_sheet[balance_sheet['index'] == 'Other Current Borrowings'].iloc[:, 1].iloc[0]
except:
    Other_Current_Borrowings = 0
try:
    Current_Notes_Payable = balance_sheet[balance_sheet['index'] == 'Current Notes Payable'].iloc[:, 1].iloc[0]
except:
    Current_Notes_Payable = 0
try:
    Payables = balance_sheet[balance_sheet['index'] == 'Payables'].iloc[:, 1].iloc[0]
except:
    Payables = 0

Accounts_Payable = balance_sheet[balance_sheet['index'] == 'Accounts Payable'].iloc[:, 1].iloc[0]

try:
    Current_Accrued_Expenses = balance_sheet[balance_sheet['index'] == 'Current Accrued Expenses'].iloc[:, 1].iloc[0]
except:
    Current_Notes_Payable = 0

try:
    Payables_And_Accrued_Expenses = \
        balance_sheet[balance_sheet['index'] == 'Payables And Accrued Expenses'].iloc[:, 1].iloc[0]
except:
    Payables_And_Accrued_Expenses = 0
Current_Liabilities = Current_Debt + Other_Current_Borrowings + Current_Notes_Payable + \
                      Payables + Accounts_Payable + Current_Accrued_Expenses + Payables_And_Accrued_Expenses
Working_Capital = Current_Assets - Current_Liabilities

Total_Revenue = financials[financials['index'] == 'Total Revenue'].iloc[:, 1].iloc[0]

Working_Capital_Turnover = round(Total_Revenue / Working_Capital, 2)

# 4. Inventory Turnover Ratio

# Calculate  Cost Revenue
try:
    Cost_of_Revenue = income_statement[income_statement['index'] == 'Cost Of Revenue'].iloc[:, 1].iloc[0]
except:
    Cost_of_Revenue = 0

# Calculate Total Inventory: Final Year
try:
    Inventory_final_yr = balance_sheet[balance_sheet['index'] == 'Inventory'].iloc[:, 1].iloc[0]
except:
    Inventory_final_yr = 0

try:
    Finished_Goods_final_yr = balance_sheet[balance_sheet['index'] == 'Finished Goods'].iloc[:, 1].iloc[0]
except:
    Finished_Goods_final_yr = 0

try:
    Raw_Materials_final_yr = balance_sheet[balance_sheet['index'] == 'Raw Materials'].iloc[:, 1].iloc[0]
except:
    Raw_Materials_final_yr = 0

Total_Inventory_final_yr = Inventory_final_yr + Finished_Goods_final_yr + Raw_Materials_final_yr

# Calculate Total Inventory : 1 year then
try:
    Inventory_1yr_then = balance_sheet[balance_sheet['index'] == 'Inventory'].iloc[:, 2].iloc[0]
except:
    Inventory_1yr_then = 0

try:
    Finished_Goods_1yr_then = balance_sheet[balance_sheet['index'] == 'Finished Goods'].iloc[:, 2].iloc[0]
except:
    Finished_Goods_1yr_then = 0

try:
    Raw_Materials_1yr_then = balance_sheet[balance_sheet['index'] == 'Raw Materials'].iloc[:, 2].iloc[0]
except:
    Raw_Materials_1yr_then = 0

Total_Inventory_1yr_then = Inventory_1yr_then + Finished_Goods_1yr_then + Raw_Materials_1yr_then

Average_Total_Inventory = (Total_Inventory_final_yr + Total_Inventory_final_yr) / 2

Inventory_Turnover_Ratio = round(Cost_of_Revenue / Average_Total_Inventory, 2)

## 5. Accounts Receivable Turnover Ratio

Accounts_Receivable_final_yr = balance_sheet[balance_sheet['index'] == 'Accounts Receivable'].iloc[:, 1].iloc[0]
Accounts_Receivable_1yr_then = balance_sheet[balance_sheet['index'] == 'Accounts Receivable'].iloc[:, 2].iloc[0]
Average_Receivables = (Accounts_Receivable_final_yr + Accounts_Receivable_1yr_then) / 2
Accounts_Receivable_Turnover_Ratio = round(Operating_Revenue / Average_Receivables, 2)

##6. Days Sales Outstanding
Days_Sales_Outstanding = round(365 / Accounts_Receivable_Turnover_Ratio, 2)


########################################################################################################################
# Piotroski Score
# Piotroski Score
def piotroski_score(ticker):
    score = 0
    # Criteria 1: Positive Net Income
    if net_income > 0:
        score += 1
    try:
        Operating_Cash_Flow = ticker.cash_flow.loc['Cash Flow From Continuing Operating Activities'].iloc[0]
    except KeyError:
        Operating_Cash_Flow = Operating_Cash_Flow = ticker.cash_flow.loc['Operating Cash Flow'].iloc[0]
    except:
        Operating_Cash_Flow = 0

    if Operating_Cash_Flow > 0:
        score += 1

        # Criteria 3: Return on Assets (ROA) Improvement
    roa_curr = financials[financials['index'] == 'Net Income'].iloc[:, 1].iloc[0] / \
               balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 1].iloc[0]
    roa_prev = financials[financials['index'] == 'Net Income'].iloc[:, 2].iloc[0] / \
               balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 2].iloc[0]
    if roa_curr > roa_prev:
        score += 1

    # Criteria 4: Operating Cash Flow > Net Income
    if Operating_Cash_Flow > financials[financials['index'] == 'Net Income'].iloc[:, 1].iloc[0]:
        score += 1

    # Criteria 5: Decline in Long-Term Debt-to-Assets Ratio
    Total_Debt_prev_yr = balance_sheet[balance_sheet["index"] == "Total Debt"].iloc[:, 2].iloc[0]
    Capital_Lease_Obligations_prev_yr = \
    balance_sheet[balance_sheet["index"] == "Capital Lease Obligations"].iloc[:, 2].iloc[0]

    try:
        Long_Term_Debt_prev_yr = balance_sheet[balance_sheet["index"] == "Long Term Debt"].iloc[:, 2].iloc[0]
    except:
        Long_Term_Debt_prev_yr = 0

    try:
        Current_Debt_And_Capital_Lease_Obligation_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Current Liabilities"].iloc[:, 2].iloc[0])
    except:
        Current_Debt_And_Capital_Lease_Obligation_prev_yr = 0
    try:
        Current_Liabilities_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Current Liabilities"].iloc[:, 2].iloc[0])
    except:
        Current_Liabilities_prev_yr = 0

    try:
        Long_Term_Deferred_Taxes_Liabilities_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Long Term Deferred Taxes Liabilities"].iloc[:, 2].iloc[0])
    except:
        Long_Term_Deferred_Taxes_Liabilities_prev_yr = 0

    try:
        Dividends_Payable_prev_yr = (balance_sheet[balance_sheet["index"] == "Dividends Payable"].iloc[:, 2].iloc[0])
    except:
        Dividends_Payable_prev_yr = 0

    try:
        Total_Tax_Payable_prev_yr = (balance_sheet[balance_sheet["index"] == "Total Tax Payable"].iloc[:, 2].iloc[0])
    except:
        Total_Tax_Payable_prev_yr = 0

    try:
        Accounts_Payable_prev_yr = (balance_sheet[balance_sheet["index"] == "Accounts Payable"].iloc[:, 2].iloc[0])
    except:
        Accounts_Payable_prev_yr = 0

    try:
        Total_Liabilities_Net_Minority_Interest_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Total Liabilities Net Minority Interest"].iloc[:, 2].iloc[0])
    except:
        Total_Liabilities_Net_Minority_Interest_prev_yr = 0

    try:
        Total_Non_Current_Liabilities_Net_Minority_Interest_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Total Non Current Liabilities Net Minority Interest"].iloc[:, 2].iloc[
            0])
    except:
        Total_Non_Current_Liabilities_Net_Minority_Interest_prev_yr = 0

    try:
        Long_Term_Provisions_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Long Term Provisions"].iloc[:, 2].iloc[0])
    except:
        Long_Term_Provisions_prev_yr = 0

    try:
        Current_Debt_prev_yr = (balance_sheet[balance_sheet["index"] == "Current Debt"].iloc[:, 2].iloc[0])
    except:
        Current_Debt_prev_yr = 0

    try:
        Derivative_Product_Liabilities_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Derivative Product Liabilities"].iloc[:, 2].iloc[0])
    except:
        Derivative_Product_Liabilities_prev_yr = 0

    try:
        Non_Current_Pension_And_Other_Postretirement_Benefit_Plans_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Non Current Pension And Other Postretirement Benefit Plans"].iloc[:,
        2].iloc[0])
    except:
        Non_Current_Pension_And_Other_Postretirement_Benefit_Plans_prev_yr = 0

    try:
        Trade_and_Other_Payables_Non_Current_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Trade and Other Payables Non Current"].iloc[:, 2].iloc[0])
    except:
        Trade_and_Other_Payables_Non_Current_prev_yr = 0

    try:
        Non_Current_Deferred_Revenue_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Non Current Deferred Revenue"].iloc[:, 2].iloc[0])
    except:
        Non_Current_Deferred_Revenue_prev_yr = 0

    try:
        Long_Term_Debt_And_Capital_Lease_Obligation_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Long Term Debt And Capital Lease Obligation"].iloc[:, 2].iloc[0])
    except:
        Long_Term_Debt_And_Capital_Lease_Obligation_prev_yr = 0

    try:
        Current_Notes_Payable_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Current Notes Payable"].iloc[:, 2].iloc[0])
    except:
        Current_Notes_Payable_prev_yr = 0

    try:
        Payables_And_Accrued_Expenses_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Payables And Accrued Expenses"].iloc[:, 2].iloc[0])
    except:
        Payables_And_Accrued_Expenses_prev_yr = 0

    try:
        Current_Accrued_Expenses_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Current Accrued Expenses"].iloc[:, 2].iloc[0])
    except:
        Current_Accrued_Expenses_prev_yr = 0

    try:
        Payables_prev_yr = (balance_sheet[balance_sheet["index"] == "Payables"].iloc[:, 2].iloc[0])
    except:
        Payables_prev_yr = 0

    Long_Term_Capital_Lease_Obligation_prev_yr = (
    balance_sheet[balance_sheet["index"] == "Long Term Capital Lease Obligation"].iloc[:, 2].iloc[0])

    try:
        Current_Capital_Lease_Obligation_prev_yr = (
        balance_sheet[balance_sheet["index"] == "Current Capital Lease Obligation"].iloc[:, 2].iloc[0])
    except:
        Current_Capital_Lease_Obligation_prev_yr = 0

    Total_Liabilities_prev_yr = Total_Debt_prev_yr + Capital_Lease_Obligations_prev_yr + Long_Term_Debt_prev_yr + Current_Debt_And_Capital_Lease_Obligation_prev_yr + Current_Liabilities_prev_yr + \
                                Long_Term_Deferred_Taxes_Liabilities_prev_yr + Dividends_Payable_prev_yr + Total_Tax_Payable_prev_yr + Accounts_Payable_prev_yr + \
                                Total_Liabilities_Net_Minority_Interest_prev_yr + Total_Non_Current_Liabilities_Net_Minority_Interest_prev_yr + \
                                Long_Term_Provisions_prev_yr + Current_Debt_prev_yr + Derivative_Product_Liabilities_prev_yr + Non_Current_Pension_And_Other_Postretirement_Benefit_Plans_prev_yr + \
                                Trade_and_Other_Payables_Non_Current_prev_yr + Non_Current_Deferred_Revenue_prev_yr + Long_Term_Debt_And_Capital_Lease_Obligation_prev_yr + \
                                Current_Notes_Payable_prev_yr + Payables_And_Accrued_Expenses_prev_yr + Current_Accrued_Expenses_prev_yr + Payables_prev_yr + Long_Term_Capital_Lease_Obligation_prev_yr + \
                                Current_Capital_Lease_Obligation_prev_yr

    assets_prev = balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 2].iloc[0]
    assets_curr = balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 1].iloc[0]

    debt_to_assets_curr = Total_Liabilities / assets_curr

    debt_to_assets_prev = Total_Liabilities / assets_prev

    if debt_to_assets_curr < debt_to_assets_prev:
        score += 1
    # Criteria 6: Increase in Current Ratio

    # Calculate Previous Year Current  Assets
    try:
        Cash_And_Cash_Equivalents_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Cash And Cash Equivalents'].iloc[:, 2].iloc[0]
    except:
        Cash_And_Cash_Equivalents_prev_yr = 0
    try:
        Other_Short_Term_Investments_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Other Short Term Investments'].iloc[:, 2].iloc[0]
    except:
        Other_Short_Term_Investments_prev_yr = 0
    try:
        Accounts_Receivable_prev_yr = balance_sheet[balance_sheet['index'] == 'Accounts Receivable'].iloc[:, 2].iloc[0]
    except:
        Accounts_Receivable_prev_yr = 0
    try:
        Other_Receivables_prev_yr = balance_sheet[balance_sheet['index'] == 'Other Receivables'].iloc[:, 2].iloc[0]
    except:
        Other_Receivables_prev_yr = 0
    try:
        Receivables_prev_yr = balance_sheet[balance_sheet['index'] == 'Receivables'].iloc[:, 2].iloc[0]
    except:
        Receivables_prev_yr = 0
    try:
        Prepaid_Assets_prev_yr = balance_sheet[balance_sheet['index'] == 'Prepaid Assets'].iloc[:, 2].iloc[0]
    except:
        Prepaid_Assets_prev_yr = 0

    Current_Assets_prev_yr = Cash_And_Cash_Equivalents_prev_yr + \
                             Other_Short_Term_Investments_prev_yr + Accounts_Receivable_prev_yr + \
                             Other_Receivables_prev_yr + Receivables_prev_yr + Prepaid_Assets_prev_yr

    # Calculate Previous Year Current Liabilities

    if 'Current Debt' in balance_sheet['index'].unique().tolist():
        Current_Debt_prev_yr = balance_sheet[balance_sheet['index'] == 'Current Debt'].iloc[:, 2].iloc[0]
    else:
        Current_Debt_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Debt And Capital Lease Obligation'].iloc[:, 2].iloc[0] + \
        balance_sheet[balance_sheet['index'] == 'Current Capital Lease Obligation'].iloc[:, 2].iloc[0]

    try:
        Other_Current_Borrowings_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Other Current Borrowings'].iloc[:, 2].iloc[0]
    except:
        Other_Current_Borrowings_prev_yr = 0
    try:
        Current_Notes_Payable_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Notes Payable'].iloc[:, 2].iloc[0]
    except:
        Current_Notes_Payable_prev_yr = 0
    try:
        Payables_prev_yr = balance_sheet[balance_sheet['index'] == 'Payables'].iloc[:, 2].iloc[0]
    except:
        Payables_prev_yr = 0

    Accounts_Payable_prev_yr = balance_sheet[balance_sheet['index'] == 'Accounts Payable'].iloc[:, 2].iloc[0]

    try:
        Current_Accrued_Expenses_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Accrued Expenses'].iloc[:, 2].iloc[0]
    except:
        Current_Notes_Payable_prev_yr = 0

    try:
        Payables_And_Accrued_Expenses_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Payables And Accrued Expenses'].iloc[:, 2].iloc[0]
    except:
        Payables_And_Accrued_Expenses_prev_yr = 0
    Current_Liabilities_prev_yr = Current_Debt_prev_yr + Other_Current_Borrowings_prev_yr + Current_Notes_Payable_prev_yr + \
                                  Payables_prev_yr + Accounts_Payable_prev_yr + Current_Accrued_Expenses_prev_yr + Payables_And_Accrued_Expenses_prev_yr

    current_ratio_curr = Current_Assets / Current_Liabilities
    current_ratio_prev = Current_Assets_prev_yr / Current_Liabilities_prev_yr
    if current_ratio_curr > current_ratio_prev:
        score += 1

    # Criteria 6: Increase in Current Ratio

    # Calculate Previous Year Current  Assets
    Cash_And_Cash_Equivalents_prev_yr = \
    balance_sheet[balance_sheet['index'] == 'Cash And Cash Equivalents'].iloc[:, 2].iloc[0]
    Other_Short_Term_Investments_prev_yr = \
    balance_sheet[balance_sheet['index'] == 'Other Short Term Investments'].iloc[:, 2].iloc[0]
    Accounts_Receivable_prev_yr = balance_sheet[balance_sheet['index'] == 'Accounts Receivable'].iloc[:, 2].iloc[0]
    Other_Receivables_prev_yr = balance_sheet[balance_sheet['index'] == 'Other Receivables'].iloc[:, 2].iloc[0]
    try:
        Receivables_prev_yr = balance_sheet[balance_sheet['index'] == 'Receivables'].iloc[:, 2].iloc[0]
    except:
        Receivables_prev_yr = 0
    try:
        Prepaid_Assets_prev_yr = balance_sheet[balance_sheet['index'] == 'Prepaid Assets'].iloc[:, 2].iloc[0]
    except:
        Prepaid_Assets = 0

    Current_Assets_prev_yr = Cash_And_Cash_Equivalents_prev_yr + \
                             Other_Short_Term_Investments_prev_yr + Accounts_Receivable_prev_yr + \
                             Other_Receivables_prev_yr + Receivables_prev_yr + Prepaid_Assets_prev_yr
    # Calculate Previous Year Current Liabilities

    if 'Current Debt' in balance_sheet['index'].unique().tolist():
        Current_Debt_prev_yr = balance_sheet[balance_sheet['index'] == 'Current Debt'].iloc[:, 2].iloc[0]
    else:
        Current_Debt_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Debt And Capital Lease Obligation'].iloc[:, 2].iloc[0] + \
        balance_sheet[balance_sheet['index'] == 'Current Capital Lease Obligation'].iloc[:, 2].iloc[0]

    try:
        Other_Current_Borrowings_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Other Current Borrowings'].iloc[:, 2].iloc[0]
    except:
        Other_Current_Borrowings_prev_yr = 0
    try:
        Current_Notes_Payable_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Notes Payable'].iloc[:, 2].iloc[0]
    except:
        Current_Notes_Payable_prev_yr = 0
    try:
        Payables_prev_yr = balance_sheet[balance_sheet['index'] == 'Payables'].iloc[:, 2].iloc[0]
    except:
        Payables_prev_yr = 0

    Accounts_Payable_prev_yr = balance_sheet[balance_sheet['index'] == 'Accounts Payable'].iloc[:, 2].iloc[0]

    try:
        Current_Accrued_Expenses_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Current Accrued Expenses'].iloc[:, 2].iloc[0]
    except:
        Current_Notes_Payable_prev_yr = 0

    try:
        Payables_And_Accrued_Expenses_prev_yr = \
        balance_sheet[balance_sheet['index'] == 'Payables And Accrued Expenses'].iloc[:, 2].iloc[0]
    except:
        Payables_And_Accrued_Expenses_prev_yr = 0
    Current_Liabilities_prev_yr = Current_Debt_prev_yr + Other_Current_Borrowings_prev_yr + Current_Notes_Payable_prev_yr + \
                                  Payables_prev_yr + Accounts_Payable_prev_yr + Current_Accrued_Expenses_prev_yr + Payables_And_Accrued_Expenses_prev_yr

    current_ratio_curr = Current_Assets / Current_Liabilities
    current_ratio_prev = Current_Assets_prev_yr / Current_Liabilities_prev_yr
    if current_ratio_curr > current_ratio_prev:
        score += 1
    # Criteria 7: No New Equity Issuance
    shares_curr = balance_sheet[balance_sheet['index'] == 'Share Issued'].iloc[:, 1].iloc[0]
    shares_prev = balance_sheet[balance_sheet['index'] == 'Share Issued'].iloc[:, 2].iloc[0]
    if shares_curr <= shares_prev:
        score += 1
    # Criteria 8: Higher Gross Margin
    # Calculate Operating Expenses Of Present Year
    try:
        Selling_General_And_Administration = \
        income_statement[income_statement['index'] == 'Selling General And Administration'].iloc[:, 1].iloc[0]
    except:
        Selling_General_And_Administration = 0
    try:
        Other_Gand_A = income_statement[income_statement['index'] == 'Other Gand A'].iloc[:, 1].iloc[0]
    except:
        Other_Gand_A = 0
    try:
        Salaries_And_Wages = income_statement[income_statement['index'] == 'Salaries And Wages'].iloc[:, 1].iloc[0]
    except:
        Salaries_And_Wages = 0
    try:
        Depreciation_Income_Statement = \
        income_statement[income_statement['index'] == 'Depreciation Income Statement'].iloc[:, 1].iloc[0]
    except:
        Depreciation_Income_Statement = 0
    try:
        Amortization_Of_Intangibles_Income_Statement = \
        income_statement[income_statement['index'] == 'Amortization Of Intangibles Income Statement'].iloc[:, 1].iloc[0]
    except:
        Amortization_Of_Intangibles_Income_Statement = 0
    try:
        depreciation_amortization_income_statement = \
        income_statement[income_statement['index'] == 'Depreciation And Amortization In Income Statement'].iloc[:,
        1].iloc[0]
    except:
        depreciation_amortization_income_statement = 0

    operating_expenses = Selling_General_And_Administration + Other_Gand_A + Salaries_And_Wages + \
                         Depreciation_Income_Statement + Amortization_Of_Intangibles_Income_Statement + \
                         Amortization_Of_Intangibles_Income_Statement + depreciation_amortization_income_statement

    # Calculate cost of goods sold Of Present Year
    cost_of_goods_sold = revenue - operating_expenses

    # Calculate gross profit Of Present Year
    gross_profit = revenue - cost_of_goods_sold
    gross_margin_curr = (gross_profit / revenue) * 100

    # Calculate Total Revenue of Previous year
    revenue_prev_yr = (income_statement[income_statement['index'] == 'Total Revenue'].iloc[:, 2].iloc[0])

    # Calculate Operating Expenses Of Previous Year
    try:
        Selling_General_And_Administration_prev_yr = \
        income_statement[income_statement['index'] == 'Selling General And Administration'].iloc[:, 2].iloc[0]
    except:
        Selling_General_And_Administration_prev_yr = 0
    try:
        Other_Gand_A_prev_yr = income_statement[income_statement['index'] == 'Other Gand A'].iloc[:, 2].iloc[0]
    except:
        Other_Gand_A_prev_yr = 0
    try:
        Salaries_And_Wages_prev_yr = \
        income_statement[income_statement['index'] == 'Salaries And Wages'].iloc[:, 2].iloc[0]
    except:
        Salaries_And_Wages_prev_yr = 0
    try:
        Depreciation_Income_Statement_prev_yr = \
        income_statement[income_statement['index'] == 'Depreciation Income Statement'].iloc[:, 2].iloc[0]
    except:
        Salaries_And_Wages_prev_yr = 0
    try:
        Amortization_Of_Intangibles_Income_Statement_prev_yr = \
        income_statement[income_statement['index'] == 'Amortization Of Intangibles Income Statement'].iloc[:, 2].iloc[0]
    except:
        Amortization_Of_Intangibles_Income_Statement_prev_yr = 0
    try:
        depreciation_amortization_income_statement_prev_yr = \
        income_statement[income_statement['index'] == 'Depreciation And Amortization In Income Statement'].iloc[:,
        2].iloc[0]
    except:
        depreciation_amortization_income_statement_prev_yr = 0

    operating_expenses_prev_yr = Selling_General_And_Administration_prev_yr + Other_Gand_A_prev_yr + Salaries_And_Wages_prev_yr + \
                                 Depreciation_Income_Statement_prev_yr + Amortization_Of_Intangibles_Income_Statement_prev_yr + \
                                 Amortization_Of_Intangibles_Income_Statement_prev_yr + depreciation_amortization_income_statement_prev_yr

    # Calculate cost of goods sold Of Present Year
    cost_of_goods_sold_prev_yr = revenue_prev_yr - operating_expenses_prev_yr

    # Calculate gross profit Of Present Year
    gross_profit_prev_yr = revenue_prev_yr - cost_of_goods_sold_prev_yr
    gross_margin_prev = (gross_profit_prev_yr / revenue_prev_yr) * 100
    if gross_margin_curr > gross_margin_prev:
        score += 1

    # Criteria 9: Higher Asset Turnover Ratio
    asset_turnover_curr = revenue / balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 1].iloc[0]
    asset_turnover_prev = revenue_prev_yr / balance_sheet[balance_sheet['index'] == 'Total Assets'].iloc[:, 2].iloc[0]
    if asset_turnover_curr > asset_turnover_prev:
        score += 1

    return score


########################################################################################################################
def intrinsic_value(ticker, growth_rate=8, margin_of_safety=0.1):
    """
    Calculates the estimated intrinsic value of a company in India using Graham's formula.

    Args:
        ticker (str): Stock ticker symbol of the company.
        growth_rate (float): Expected annual growth rate in earnings per share (decimal).
        margin_of_safety (float, optional): Percentage discount to apply to the intrinsic value (default: 0.2).

    Returns:
        float: Estimated intrinsic value per share.
    """
    # Get current EPS
    eps = ticker.info['trailingEps']

    # Need to find the current AAA corporate bond yield in India (replace with your source)
    # Example: Assuming AAA bond yield is 7%
    bond_yield = 8.3

    # Calculate multiplier
    multiplier = 7 + growth_rate

    # Apply adjusted Graham's formula
    intrinsic_value = (eps * multiplier * 7) / bond_yield

    # Apply margin of safety
    intrinsic_value *= (1 - margin_of_safety)

    return round(intrinsic_value, 2)
########################################################################################################################

col1, col2, col3 = st.columns(3)
if 'Profitability Ratios' in columns_to_show:
    with col1:
        st.write(f"**PAT Margin** : {pat_margin} %")
        st.write(f"**PAT Growth(3 yrs)** : {cagr_pat} %")
        st.write(f"**EBITDA Margin** : {EBITDA_Margin} %")
        st.write(f"**EBITDA Growth(3 yrs)** : {cagr_EBITDA}")
        st.write(f"**ROE** : {ROE}")
        st.write(f"**ROA** :{ROA}")
        st.write(f"**ROCE** :{ROCE}")
if 'Piotroski Score' in columns_to_show:
    with col1:
        st.write(f"**Piotroski score** : {piotroski_score(ticker)}")
if "Intrinsic Value" in columns_to_show:
    with col1:
        st.write(f"**Intrinsic Value** : {intrinsic_value(ticker)}")

if "Leverage Ratios" in columns_to_show:
    with col2:
        st.write(f"**Interest Coverage ratio** : {Interest_Coverage_ratio}")
        st.write(f"**Debt to Equity ratio** : {Debt_to_Equity_ratio}")
        st.write(f"**Debt to Asset ratio** : {Debt_to_Asset_ratio}")

if 'Valuation Ratios' in columns_to_show:
    with col3:
        st.write(f"**P/E Ratio** : {PE_Ratio}")
        st.write(f"**P/BV Ratio** : {P_BV_Ratio}")
        st.write(f"**P/S Ratio** : {P_Sales_Ratio}")

if 'Operating Ratios' in columns_to_show:
    with col2:
        st.write(f"**Fixed Assets Turnover Ratio** :  {Fixed_Assets_Turnover_Ratio}")
        st.write(f"**Total Assets Turnover Ratio** : {Total_Assets_Turnover_Ratio}")
        st.write(f"**Working Capital Turnover Ratio** : {Working_Capital_Turnover}")
        st.write(f"**Inventory Turnover Ratio** : {Inventory_Turnover_Ratio}")
        st.write(f"**Accounts Receivable Turnover Ratio**: {Accounts_Receivable_Turnover_Ratio}")
        st.write(f"**Days Sales Outstanding**: {Days_Sales_Outstanding}")
