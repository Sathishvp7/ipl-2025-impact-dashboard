# Working code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import re
import os
import shutil

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Define the data directory
DATA_DIR = Path('Dataset')

def load_all_data():
    """Load all Excel files from the Dataset directory"""
    print("Loading all data files...")
    
    # Load advertisers data
    advertisers_path = DATA_DIR / 'fact_ipl_advertisers.xlsx'
    advertisers_df = pd.read_excel(advertisers_path)
    print(f"Loaded {advertisers_path.name} with {len(advertisers_df)} rows")
    
    # Load central contracts data
    contracts_path = DATA_DIR / 'fact_ipl_central_contracts.xlsx'
    contracts_df = pd.read_excel(contracts_path)
    print(f"Loaded {contracts_path.name} with {len(contracts_df)} rows")
    
    # Load revenue demography data
    revenue_path = DATA_DIR / 'fact_revenue_demography.xlsm'
    revenue_df = pd.read_excel(revenue_path)
    print(f"Loaded {revenue_path.name} with {len(revenue_df)} rows")
    
    # Load summary demography data
    summary_path = DATA_DIR / 'fact_summary_demography.xlsx'
    summary_df = pd.read_excel(summary_path)
    print(f"Loaded {summary_path.name} with {len(summary_df)} rows")
    
    return {
        'advertisers': advertisers_df,
        'contracts': contracts_df,
        'revenue': revenue_df,
        'summary': summary_df
    }

def clean_data(dataframes):
    """Clean and preprocess the data"""
    print("Cleaning and processing data...")
    
    # Create copies to avoid modifying original dataframes
    dfs = {k: v.copy() for k, v in dataframes.items()}
    
    # Clean advertisers data
    if 'advertisers' in dfs:
        # Convert column names to lowercase and replace spaces with underscores
        dfs['advertisers'].columns = [col.lower().replace(' ', '_') for col in dfs['advertisers'].columns]
        
    # Clean contracts data
    if 'contracts' in dfs:
        # Convert column names to lowercase and replace spaces with underscores
        dfs['contracts'].columns = [col.lower().replace(' ', '_') for col in dfs['contracts'].columns]
        
    # Clean revenue data
    if 'revenue' in dfs:
        # Convert column names to lowercase and replace spaces with underscores
        dfs['revenue'].columns = [col.lower().replace(' ', '_') for col in dfs['revenue'].columns]
        
        # Clean latest_annual_revenue column if it exists
        if 'latest_annual_revenue' in dfs['revenue'].columns:
            # Function to extract first numeric value from complex string
            def extract_first_number(value):
                if pd.isna(value):
                    return 0
                
                # Convert to string if not already
                value_str = str(value)
                
                # Extract all numbers from the string
                numbers = re.findall(r'[\d,]+(?:\.\d+)?', value_str)
                
                # If we found numbers, return the first one, removing commas
                if numbers:
                    return float(numbers[0].replace(',', ''))
                
                return 0
            
            # Apply the extraction function
            dfs['revenue']['numeric_revenue'] = dfs['revenue']['latest_annual_revenue'].apply(extract_first_number)
    
    # Clean summary data
    if 'summary' in dfs:
        # Convert column names to lowercase and replace spaces with underscores
        dfs['summary'].columns = [col.lower().replace(' ', '_') for col in dfs['summary'].columns]
        
        # Extract numeric values from estimated_user_population if it exists
        if 'estimated_user_population' in dfs['summary'].columns:
            def extract_avg_population(value):
                if pd.isna(value):
                    return 0
                
                # Convert to string if not already
                value_str = str(value)
                
                # Extract all numbers from the string
                numbers = re.findall(r'[\d.]+', value_str)
                
                # If we found at least two numbers, compute average
                if len(numbers) >= 2:
                    try:
                        return (float(numbers[0]) + float(numbers[1])) / 2
                    except (ValueError, IndexError):
                        pass
                
                # If we found at least one number, return that
                if numbers:
                    try:
                        return float(numbers[0])
                    except (ValueError, IndexError):
                        pass
                
                return 0
            
            dfs['summary']['avg_user_population'] = dfs['summary']['estimated_user_population'].apply(extract_avg_population)
    
    # Standardize income group categories
    for df_name in ['revenue', 'summary']:
        if df_name in dfs and 'income_group' in dfs[df_name].columns:
            # Function to standardize income group values
            def standardize_income_group(income_value):
                if pd.isna(income_value):
                    return "Unknown"
                
                # Convert to string and lowercase for consistent comparison
                income_str = str(income_value).lower().strip()
                
                # Define mappings from various formats to standardized categories
                income_mappings = {
                    'lower': 'Lower Income',
                    'lower income': 'Lower Income',
                    'lower-middle': 'Lower-Middle',
                    'lower middle': 'Lower-Middle',
                    'middle': 'Middle',
                    'upper-middle': 'Upper-Middle',
                    'upper middle': 'Upper-Middle',
                    'upper': 'Upper Income',
                    'high': 'Upper Income',
                    'upper income': 'Upper Income',
                    'high income': 'Upper Income'
                }
                
                # For complex multi-category strings
                if '&' in income_str or ',' in income_str:
                    parts = re.split(r'[,&]', income_str)
                    standard_parts = []
                    for part in parts:
                        clean_part = part.strip()
                        for key, value in income_mappings.items():
                            if key in clean_part:
                                standard_parts.append(value)
                                break
                        else:
                            if clean_part:
                                standard_parts.append(clean_part.title())
                    
                    return ', '.join(sorted(set(standard_parts)))
                
                # For simple single-category strings
                for key, value in income_mappings.items():
                    if key in income_str:
                        return value
                
                return income_str.title()
            
            dfs[df_name]['standardized_income_group'] = dfs[df_name]['income_group'].apply(standardize_income_group)
    
    return dfs

def analyze_economic_impact(dataframes):
    """Analyze the economic impact of IPL"""
    print("Analyzing economic impact...")
    
    contracts_df = dataframes['contracts']
    revenue_df = dataframes['revenue']
    
    # Analyze contract values
    total_sponsorship = contracts_df['amount_in_crores_2025'].sum() if 'amount_in_crores_2025' in contracts_df.columns else 0
    
    # Create contract type with sponsor name in brackets
    if 'contract_type' in contracts_df.columns and 'amount_in_crores_2025' in contracts_df.columns:
        # Check if partner_sponsor_name or other similar column exists
        partner_col = None
        if 'partner_sponsor_name' in contracts_df.columns:
            partner_col = 'partner_sponsor_name'
        elif 'sponsor_name' in contracts_df.columns:
            partner_col = 'sponsor_name'
        elif 'company' in contracts_df.columns:
            partner_col = 'company'
            
        if partner_col:
            # Create a new column with contract_type (partner_name)
            contracts_df['contract_type_with_partner'] = contracts_df.apply(
                lambda row: f"{row['contract_type']} ({row[partner_col]})" if not pd.isna(row[partner_col]) else row['contract_type'],
                axis=1
            )
            # Group by the new column
            contract_by_type = contracts_df.groupby('contract_type_with_partner')['amount_in_crores_2025'].sum()
        else:
            # Fallback to just contract_type if no partner column is found
            contract_by_type = contracts_df.groupby('contract_type')['amount_in_crores_2025'].sum()
    else:
        contract_by_type = pd.Series()
    
    # Analyze revenue impact by company sector
    if 'sector' in revenue_df.columns and 'numeric_revenue' in revenue_df.columns:
        sector_revenue = revenue_df.groupby('sector')['numeric_revenue'].sum()
    else:
        sector_revenue = pd.Series()
    
    # Income group impact analysis
    if 'standardized_income_group' in revenue_df.columns and 'numeric_revenue' in revenue_df.columns:
        income_group_revenue = revenue_df.groupby('standardized_income_group')['numeric_revenue'].sum()
    else:
        income_group_revenue = pd.Series()
    
    # Enhanced income group analysis
    
    # 1. Count companies targeting each income group
    companies_by_income = pd.Series()
    companies_list_by_income = {}  # Dictionary to store list of companies by income group
    
    if 'standardized_income_group' in revenue_df.columns:
        # Split multi-value income groups and explode the dataframe
        def explode_income_groups(df):
            df_copy = df.copy()
            if 'standardized_income_group' not in df_copy.columns:
                return df_copy
            
            df_copy['standardized_income_group'] = df_copy['standardized_income_group'].astype(str)
            df_copy['income_group_list'] = df_copy['standardized_income_group'].str.split(',')
            df_exploded = df_copy.explode('income_group_list')
            df_exploded['income_group_list'] = df_exploded['income_group_list'].str.strip()
            
            return df_exploded
        
        revenue_exploded = explode_income_groups(revenue_df)
        companies_by_income = revenue_exploded['income_group_list'].value_counts().sort_index()
        
        # Get company names for each income group
        if 'company' in revenue_exploded.columns:
            for income_group in companies_by_income.index:
                # Make sure the income group is valid
                if pd.notna(income_group) and income_group != '':
                    # Get companies for this income group and convert to a list, ensuring no NaN values
                    companies = revenue_exploded[revenue_exploded['income_group_list'] == income_group]['company']
                    companies = companies[companies.notna()].unique().tolist()
                    companies_list_by_income[income_group] = companies
    
    # 2. Calculate average revenue by income group
    avg_revenue_by_income = pd.Series()
    if 'numeric_revenue' in revenue_df.columns and len(companies_by_income) > 0:
        avg_revenue_by_income = revenue_exploded.groupby('income_group_list')['numeric_revenue'].mean().sort_index()
    
    # 3. Analyze sector distribution by income group
    sector_by_income = pd.DataFrame()
    if 'sector' in revenue_df.columns and len(companies_by_income) > 0:
        sector_by_income = revenue_exploded.groupby(['income_group_list', 'sector']).size().unstack(fill_value=0)
    
    # 4. Age group and income group relationship
    age_income_crosstab = pd.DataFrame()
    if 'age_group' in revenue_df.columns and 'standardized_income_group' in revenue_df.columns:
        # Need to handle both string and non-string age_group values
        age_col = revenue_df['age_group'].str.strip() if isinstance(revenue_df['age_group'].iloc[0], str) else revenue_df['age_group']
        age_income_crosstab = pd.crosstab(revenue_df['standardized_income_group'].str.strip(), age_col)
    
    return {
        'total_sponsorship': total_sponsorship,
        'contract_by_type': contract_by_type,
        'sector_revenue': sector_revenue,
        'income_group_revenue': income_group_revenue,
        'companies_by_income': companies_by_income,
        'companies_list_by_income': companies_list_by_income,  # Add the companies list
        'avg_revenue_by_income': avg_revenue_by_income,
        'sector_by_income': sector_by_income,
        'age_income_crosstab': age_income_crosstab
    }

def analyze_social_impact(dataframes):
    """Analyze the social and health implications of IPL advertising"""
    print("Analyzing social impact...")
    
    advertisers_df = dataframes['advertisers']
    revenue_df = dataframes['revenue']
    summary_df = dataframes['summary']
    
    # Analyze health and social risks
    risk_by_category = advertisers_df.groupby('category')['health_social_risk'].value_counts() if 'category' in advertisers_df.columns and 'health_social_risk' in advertisers_df.columns else pd.Series()
    
    # Create a dictionary to store advertisers by category and risk level
    brands_by_category_risk = {}
    if 'category' in advertisers_df.columns and 'health_social_risk' in advertisers_df.columns and 'advertiser_brand' in advertisers_df.columns:
        # Group data by category and risk level
        for (category, risk), group in advertisers_df.groupby(['category', 'health_social_risk']):
            # Create a unique key for this category-risk combination
            key = f"{category}_{risk}"
            # Get unique advertiser brands
            if 'advertiser_brand' in group.columns:
                brands = group['advertiser_brand'].dropna().unique().tolist()
                brands_by_category_risk[key] = brands
    
    # Analyze celebrity influence on potentially harmful products
    celebrity_impact = advertisers_df.groupby(['celebrity_influence', 'health_social_risk']).size() if 'celebrity_influence' in advertisers_df.columns and 'health_social_risk' in advertisers_df.columns else pd.Series()

    # Analyze ad type distribution
    ad_type_dist = advertisers_df['category'].value_counts() if 'category' in advertisers_df.columns else pd.Series()
    
    return {
        'risk_by_category': risk_by_category,
        'brands_by_category_risk': brands_by_category_risk,  # Add brands dictionary
        'celebrity_impact': celebrity_impact,
        'ad_type_dist': ad_type_dist
    }

def create_plotly_figures(economic_data, social_data):
    """Create Plotly figures for the dashboard"""
    figures = {}
    
    # Economic Impact Figures
    
    # 1. Contract by Type Bar Chart
    if hasattr(economic_data['contract_by_type'], 'sort_values'):
        sorted_contract = economic_data['contract_by_type'].sort_values(ascending=False)
        
        # Create a DataFrame for more custom hover information
        contract_df = pd.DataFrame({
            'contract_type': sorted_contract.index,
            'amount': sorted_contract.values
        })
        
        # Extract contract type and partner name
        contract_df[['contract_type_clean', 'partner_name']] = contract_df['contract_type'].str.extract(r'(.*?)\s*(?:\((.*?)\))?$')
        
        # Fill NaN partner names
        contract_df['partner_name'] = contract_df['partner_name'].fillna('Unknown')
        
        # Create custom hover text
        contract_df['hover_text'] = contract_df.apply(
            lambda row: f"<b>Contract Type:</b> {row['contract_type_clean']}<br>" +
                        f"<b>Partner:</b> {row['partner_name']}<br>" +
                        f"<b>Amount:</b> ₹{row['amount']:.1f} Cr",
            axis=1
        )
        
        # Create the bar chart with enhanced hover
        figures['contract_by_type'] = px.bar(
            contract_df,
            x='contract_type', 
            y='amount',
            title='IPL 2025 Sponsorship Amount by Contract Type and Partner',
            labels={'contract_type': 'Contract Type (Partner/Sponsor)', 'amount': 'Amount in Crores (₹)'},
            color='amount',
            color_continuous_scale='Viridis',
            text=contract_df['amount'].round(1),
            hover_data={'amount': False, 'contract_type': False},  # Hide default hover
            custom_data=['hover_text', 'partner_name', 'contract_type_clean']  # Add custom hover data
        )
        
        # Update traces for improved appearance
        figures['contract_by_type'].update_traces(
            texttemplate='₹%{text} Cr',
            textposition='outside',
            hovertemplate='%{customdata[0]}'  # Use the first custom data (hover_text)
        )
        
        # Update layout with color-coded partner names in axis labels
        figures['contract_by_type'].update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
            height=650,  # Increase overall height
            margin=dict(b=150, t=80, r=50),  # Add more top and right margin to prevent cut-off
            xaxis_tickformat='',  # Remove default formatting
        )
    
    # 2. Sector Revenue Bar Chart
    if hasattr(economic_data['sector_revenue'], 'sort_values'):
        sorted_sector = economic_data['sector_revenue'].sort_values(ascending=False)
        figures['sector_revenue'] = px.bar(
            x=sorted_sector.index, 
            y=sorted_sector.values,
            title='Revenue by Sector',
            labels={'x': 'Sector', 'y': 'Revenue (Crores ₹)'},
            color=sorted_sector.values,
            color_continuous_scale='Viridis',
            text=sorted_sector.values.round(1)
        )
        figures['sector_revenue'].update_traces(
            texttemplate='₹%{text} Cr',
            textposition='outside'
        )
        figures['sector_revenue'].update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 3. Income Group Revenue Bar Chart
    if hasattr(economic_data['income_group_revenue'], 'sort_values'):
        sorted_income = economic_data['income_group_revenue'].sort_values(ascending=False)
        figures['income_group_revenue'] = px.bar(
            x=sorted_income.index, 
            y=sorted_income.values,
            title='Revenue Impact by Income Group',
            labels={'x': 'Income Group', 'y': 'Revenue (Crores ₹)'},
            color=sorted_income.values,
            color_continuous_scale='Viridis',
            text=sorted_income.values.round(1)
        )
        figures['income_group_revenue'].update_traces(
            texttemplate='₹%{text} Cr',
            textposition='outside'
        )
        figures['income_group_revenue'].update_layout(
            coloraxis_showscale=False,
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 4. Companies by Income Group
    if hasattr(economic_data['companies_by_income'], 'sort_values'):
        # Create a DataFrame for better hover information
        income_df = pd.DataFrame({
            'income_group': economic_data['companies_by_income'].index,
            'count': economic_data['companies_by_income'].values
        })
        
        # Create a safe mapping of income groups to company lists
        company_map = {}
        for k, v in economic_data.get('companies_list_by_income', {}).items():
            if isinstance(v, list):
                company_map[k] = v
            else:
                company_map[k] = []
        
        # Add companies list for hover info, ensuring we always get a list
        income_df['companies_list'] = income_df['income_group'].apply(
            lambda x: company_map.get(x, [])
        )
        
        # Format the companies list for hover display with line breaks
        def format_companies_list(companies):
            # Ensure companies is a list
            if not isinstance(companies, list):
                return "No company data available"
            
            # If the list is empty
            if len(companies) == 0:
                return "No company data available"
            
            # Limit to first 10 companies if list is too long
            if len(companies) > 10:
                return "<br>".join(companies[:10]) + f"<br>+{len(companies)-10} more"
            else:
                return "<br>".join(companies)
        
        income_df['companies_display'] = income_df['companies_list'].apply(format_companies_list)
        
        # Create custom hover text
        income_df['hover_text'] = income_df.apply(
            lambda row: f"<b>{row['income_group']}</b><br>" +
                        f"Number of Companies: <b>{row['count']}</b><br><br>" +
                        f"<b>Companies targeting this segment:</b><br>" +
                        f"{row['companies_display']}",
            axis=1
        )
        
        figures['companies_by_income'] = px.bar(
            income_df,
            x='income_group',
            y='count',
            title='Number of Companies Targeting Each Income Group',
            labels={'income_group': 'Income Group', 'count': 'Number of Companies'},
            color='count',
            color_continuous_scale='Blues',
            text='count',
            hover_data={'income_group': False, 'count': False},  # Hide default hover
            custom_data=['hover_text']  # Add custom hover data
        )
        
        figures['companies_by_income'].update_traces(
            textposition='outside',
            hovertemplate='%{customdata[0]}'  # Use the custom hover text
        )
        
        figures['companies_by_income'].update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 5. Average Revenue by Income Group
    if hasattr(economic_data['avg_revenue_by_income'], 'sort_values'):
        figures['avg_revenue_by_income'] = px.bar(
            x=economic_data['avg_revenue_by_income'].index,
            y=economic_data['avg_revenue_by_income'].values,
            title='Average Company Revenue by Target Income Group',
            labels={'x': 'Income Group', 'y': 'Average Revenue (Crores ₹)'},
            color=economic_data['avg_revenue_by_income'].values,
            color_continuous_scale='Greens',
            text=economic_data['avg_revenue_by_income'].values.round(1)
        )
        figures['avg_revenue_by_income'].update_traces(
            texttemplate='₹%{text} Cr',
            textposition='outside'
        )
        figures['avg_revenue_by_income'].update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 6. Sector Distribution by Income Group
    if not economic_data['sector_by_income'].empty:
        # Convert to long format for plotly
        sector_income_df = economic_data['sector_by_income'].reset_index().melt(
            id_vars='income_group_list',
            var_name='sector',
            value_name='count'
        )
        
        figures['sector_by_income'] = px.bar(
            sector_income_df,
            x='income_group_list',
            y='count',
            color='sector',
            title='Sector Distribution by Target Income Group',
            labels={
                'income_group_list': 'Income Group',
                'count': 'Number of Companies',
                'sector': 'Sector'
            },
            barmode='stack'
        )
        figures['sector_by_income'].update_layout(
            xaxis_tickangle=-45,
            legend_title='Sector',
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 7. Age-Income Relationship Heatmap
    if not economic_data['age_income_crosstab'].empty:
        heatmap_data = economic_data['age_income_crosstab']
        
        figures['age_income_heatmap'] = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlGnBu',
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        figures['age_income_heatmap'].update_layout(
            title='Relationship Between Target Age Groups and Income Groups',
            xaxis_title='Age Group',
            yaxis_title='Income Group',
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # Social Impact Figures
    
    # 1. Health and Social Risk by Category
    if hasattr(social_data['risk_by_category'], 'unstack'):
        risk_data = social_data['risk_by_category'].unstack().fillna(0)
        risk_df = risk_data.reset_index()
        risk_df_melted = pd.melt(
            risk_df, 
            id_vars='category',
            var_name='risk_level',
            value_name='count'
        )
        
        # Add brands information to the melted dataframe
        risk_df_melted['category_risk_key'] = risk_df_melted.apply(
            lambda row: f"{row['category']}_{row['risk_level']}", axis=1
        )
        
        # Create a function to get brands for a category-risk combination
        def get_brands_for_category_risk(key):
            return social_data.get('brands_by_category_risk', {}).get(key, [])
        
        # Add brands list to each row
        risk_df_melted['brands_list'] = risk_df_melted['category_risk_key'].apply(get_brands_for_category_risk)
        
        # Format brands list for hover display
        def format_brands_list(brands):
            if not isinstance(brands, list) or len(brands) == 0:
                return "No brand data available"
            
            # Limit to first 8 brands if list is too long
            if len(brands) > 8:
                return "<br>".join(brands[:8]) + f"<br>+{len(brands)-8} more"
            else:
                return "<br>".join(brands)
        
        risk_df_melted['brands_display'] = risk_df_melted['brands_list'].apply(format_brands_list)
        
        # Create custom hover text
        risk_df_melted['hover_text'] = risk_df_melted.apply(
            lambda row: f"<b>Category:</b> {row['category']}<br>" +
                        f"<b>Risk Level:</b> {row['risk_level']}<br>" +
                        f"<b>Count:</b> {row['count']}<br><br>" +
                        f"<b>Advertiser Brands:</b><br>{row['brands_display']}",
            axis=1
        )
        
        figures['risk_by_category'] = px.bar(
            risk_df_melted, 
            x='category', 
            y='count', 
            color='risk_level',
            title='Health and Social Risks by Advertisement Category',
            labels={'category': 'Advertisement Category', 'count': 'Number of Advertisements', 'risk_level': 'Risk Level'},
            barmode='stack',
            custom_data=['hover_text']  # Add custom hover data
        )
        
        figures['risk_by_category'].update_traces(
            hovertemplate='%{customdata[0]}',  # Use custom hover template
        )
        
        figures['risk_by_category'].update_layout(
            xaxis_tickangle=-45,
            legend_title='Risk Level',
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 2. Celebrity Influence Impact
    if hasattr(social_data['celebrity_impact'], 'unstack'):
        celeb_data = social_data['celebrity_impact'].unstack().fillna(0)
        celeb_df = celeb_data.reset_index()
        celeb_df_melted = pd.melt(
            celeb_df, 
            id_vars='celebrity_influence',
            var_name='risk_level',
            value_name='count'
        )
        
        figures['celebrity_impact'] = px.bar(
            celeb_df_melted, 
            x='celebrity_influence', 
            y='count', 
            color='risk_level',
            title='Celebrity Influence on Potentially Harmful Products',
            labels={
                'celebrity_influence': 'Celebrity Influence Level', 
                'count': 'Number of Advertisements', 
                'risk_level': 'Health/Social Risk'
            },
            barmode='stack'
        )
        figures['celebrity_impact'].update_layout(
            xaxis_tickangle=0,
            legend_title='Risk Level',
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 3. Advertisement Type Distribution
    if not social_data['ad_type_dist'].empty:
        figures['ad_type_dist'] = px.pie(
            names=social_data['ad_type_dist'].index,
            values=social_data['ad_type_dist'].values,
            title='Distribution of Advertisement Types in IPL 2025',
            hole=0.4,  # Create a donut chart
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        figures['ad_type_dist'].update_traces(
            textinfo='percent+label',
            pull=[0.2 if x in ['Pan Masala', 'Fantasy Gaming'] else 0 for x in social_data['ad_type_dist'].index]
        )
        figures['ad_type_dist'].update_layout(
            height=600,  # Increase height
            margin=dict(t=80, r=50)  # Add top and right margin to prevent cut-off
        )
    
    # 5. Combined Economic-Social Impact Figure
    # Create a scatterplot that shows both economic impact (revenue) and social risk
    
    return figures

def create_dashboard(figures):
    """Create a Dash dashboard with the plotly figures"""
    # Create assets folder for static files if it doesn't exist
    assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    if not os.path.exists(assets_folder):
        os.makedirs(assets_folder)
    
    # Copy logo to assets folder if it doesn't exist there
    logo_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'business_logo.png')
    logo_destination = os.path.join(assets_folder, 'business_logo.png')
    if os.path.exists(logo_source) and not os.path.exists(logo_destination):
        shutil.copy2(logo_source, logo_destination)
    
    app = dash.Dash(__name__, external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
    ])
    
    # For Heroku deployment
    server = app.server
    
    # Dashboard layout
    app.layout = html.Div([
        # Header
        html.Div([
            # Logo and title container
            html.Div([
                # Logo
                html.Img(
                    src='/assets/business_logo.png',
                    alt="Business Basics Logo",
                    style={
                        'height': '150px',             # Increased height
                        'width': '200px',              # Optional: add width for proportional scaling
                        'margin-right': '40px',
                        'border-radius': '10px',
                        'box-shadow': '0 4px 10px rgba(0,0,0,0.15)'  # Slightly stronger shadow for larger size
                    }),
                # Title and subtitle
                html.Div([
                    html.H1("IPL 2025 Economic & Social Impact Analysis", style={'margin-bottom': '5px'}),
                    html.H3("An in-depth analysis by Business Basics", style={'margin-top': '0px', 'margin-bottom': '5px'}),
                    html.P("Chief Editor: Tony Sharma | Research: Sathish Anantharaj", style={'margin-top': '0px'})
                ])
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
        ], className="header"),
        
        # Tabs for different sections
        dcc.Tabs([
            # Economic Impact Tab
            dcc.Tab(label="Economic Impact", children=[
                html.Div([
                    html.H2("Economic Impact of IPL 2025"),
                    html.P("The Indian Premier League (IPL) has become a major economic force, generating billions in revenue across various sectors."),
                    
                    # Sponsorship Data
                    html.Div([
                        html.H3("Sponsorship & Contract Analysis"),
                        html.Div([
                            dcc.Graph(
                                id='contract-type-graph',
                                figure=figures.get('contract_by_type', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                    
                    # Sector Revenue
                    html.Div([
                        html.H3("Revenue Impact by Sector"),
                        html.Div([
                            dcc.Graph(
                                id='sector-revenue-graph',
                                figure=figures.get('sector_revenue', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                    
                    # Income Group Revenue
                    html.Div([
                        html.H3("Revenue Impact by Income Group"),
                        html.Div([
                            dcc.Graph(
                                id='income-revenue-graph',
                                figure=figures.get('income_group_revenue', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                ])
            ]),
            
            # New Income Group Analysis Tab
            dcc.Tab(label="Income Group Analysis", children=[
                html.Div([
                    html.H2("Detailed Income Group Analysis"),
                    html.P("Understanding how different income groups interact with IPL content and advertisements provides valuable insights for stakeholders."),
                    
                    # Companies by Income Group
                    html.Div([
                        html.H3("Companies Targeting Each Income Group"),
                        html.Div([
                            dcc.Graph(
                                id='companies-income-graph',
                                figure=figures.get('companies_by_income', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                    
                    # Average Revenue by Income Group
                    html.Div([
                        html.H3("Average Company Revenue by Target Income Group"),
                        html.Div([
                            dcc.Graph(
                                id='avg-revenue-income-graph',
                                figure=figures.get('avg_revenue_by_income', go.Figure())
                            )
                        ], className="graph-container"),
                        html.P("This visualization shows which income groups are targeted by companies with higher average revenue, potentially indicating where advertising budgets are most concentrated.")
                    ], className="section"),
                    
                    # Sector Distribution by Income Group
                    html.Div([
                        html.H3("Sector Distribution by Target Income Group"),
                        html.Div([
                            dcc.Graph(
                                id='sector-income-graph',
                                figure=figures.get('sector_by_income', go.Figure())
                            )
                        ], className="graph-container"),
                        html.P("Different sectors target different income groups. This analysis reveals which sectors focus their marketing efforts on specific income demographics.")
                    ], className="section"),
                    
                    # Age-Income Relationship
                    html.Div([
                        html.H3("Age Group and Income Group Relationship"),
                        html.Div([
                            dcc.Graph(
                                id='age-income-heatmap',
                                figure=figures.get('age_income_heatmap', go.Figure())
                            )
                        ], className="graph-container"),
                        html.P("The heatmap visualizes the intersection of age groups and income groups, revealing demographic targeting patterns in IPL advertising.")
                    ], className="section"),
                    
                    # Key Insights about Income Groups
                    html.Div([
                        html.H3("Key Insights on Income Group Targeting"),
                        html.Ul([
                            html.Li("Lower and lower-middle income groups represent the largest viewer demographic for IPL content."),
                            html.Li("Fantasy gaming apps disproportionately target lower-income groups, raising concerns about financial vulnerability."),
                            html.Li("Upper income groups are targeted by premium brands but constitute a smaller portion of the overall IPL audience."),
                            html.Li("Middle-income viewers have the most diverse range of advertisements targeted at them across multiple sectors.")
                        ])
                    ], className="section")
                ])
            ]),
            
            # Social Impact Tab
            dcc.Tab(label="Social & Health Implications", children=[
                html.Div([
                    html.H2("Social & Health Implications of IPL Advertising"),
                    html.P("While the IPL drives economic growth, certain advertising practices raise concerns about social responsibility and public health."),
                    
                    # Health Risks by Ad Category
                    html.Div([
                        html.H3("Health and Social Risks by Advertisement Category"),
                        html.Div([
                            dcc.Graph(
                                id='risk-category-graph',
                                figure=figures.get('risk_by_category', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                    
                    # Celebrity Influence
                    html.Div([
                        html.H3("Celebrity Influence on Potentially Harmful Products"),
                        html.Div([
                            dcc.Graph(
                                id='celebrity-impact-graph',
                                figure=figures.get('celebrity_impact', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section"),
                    
                    # Ad Type Distribution
                    html.Div([
                        html.H3("Distribution of Advertisement Types"),
                        html.Div([
                            dcc.Graph(
                                id='ad-distribution-graph',
                                figure=figures.get('ad_type_dist', go.Figure())
                            )
                        ], className="graph-container")
                    ], className="section")
                ])
            ]),
            
            # Insights & Recommendations Tab
            dcc.Tab(label="Insights & Recommendations", children=[
                html.Div([
                    html.H2("Key Insights & Recommendations"),
                    
                    # Economic Insights
                    html.Div([
                        html.H3("Economic Insights"),
                        html.Ul([
                            html.Li("IPL 2025 continues to be a major economic driver, with significant revenue generation across multiple sectors."),
                            html.Li("Fantasy gaming and Pan Masala companies constitute over 50% of the advertising space, indicating their dominant role in IPL sponsorship."),
                            html.Li("Lower and middle-income groups are significant consumers of IPL content, representing a major market segment for advertisers.")
                        ])
                    ], className="section"),
                    
                    # Social Responsibility Insights
                    html.Div([
                        html.H3("Social Responsibility Concerns"),
                        html.Ul([
                            html.Li("High-risk product advertisements (pan masala, fantasy gaming) disproportionately target lower-income groups who may be more vulnerable to influence."),
                            html.Li("Celebrity endorsements significantly increase the appeal of potentially harmful products, raising ethical questions."),
                            html.Li("Current advertising trends may contribute to public health concerns and gambling-related issues.")
                        ])
                    ], className="section"),
                    
                    # Recommendations
                    html.Div([
                        html.H3("Recommendations for Balanced Growth"),
                        html.Ul([
                            html.Li("Implement stronger advertising standards for potentially harmful products during IPL broadcasts."),
                            html.Li("Encourage greater diversity in sponsorship to reduce dependence on high-risk product categories."),
                            html.Li("Develop targeted awareness campaigns about responsible consumption alongside advertisements for high-risk products."),
                            html.Li("Create incentives for sponsorship from sectors with positive social impact (education, health, technology).")
                        ])
                    ], className="section")
                ])
            ])
        ])
    ], className="dashboard-container")
    
    # Add custom CSS for styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .dashboard-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 2px solid #007bff;
                    margin-bottom: 20px;
                }
                .header h1 {
                    color: #007bff;
                    margin: 0;
                }
                .header h3 {
                    color: #555;
                    margin: 10px 0 0 0;
                }
                .header p {
                    color: #777;
                    margin: 5px 0 0 0;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }
                .section h3 {
                    color: #007bff;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                }
                .graph-container {
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.05);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    return app

def generate_report_markdown():
    """Generate a comprehensive markdown report"""
    report = """# IPL 2025 Impact Analysis: Economic Footprint & Social Implications

## Executive Summary

The Indian Premier League (IPL) 2025 represents a significant economic force in India's business landscape while also raising important questions about social responsibility and public health. This analysis examines both dimensions to provide a balanced understanding of IPL's dual impact.

### Key Findings:

1. **Economic Impact**:
   - IPL 2025 continues to drive substantial revenue across multiple sectors
   - Fantasy gaming and Pan Masala companies dominate advertising space
   - Significant economic benefits flow to various stakeholders including broadcasters, teams, and players

2. **Social & Health Implications**:
   - Over 50% of advertisements promote fantasy gaming and pan masala products
   - Lower-income groups show higher vulnerability to potentially harmful advertising
   - Celebrity endorsements significantly amplify the influence of high-risk product advertisements

## Methodology

This analysis combines data from multiple sources:
- Sponsorship and contract records from IPL 2025
- Advertising category distribution and celebrity influence metrics
- Income group demographics and susceptibility analysis
- Health and social risk assessments for product categories

## Detailed Findings

### Economic Impact Analysis

The IPL 2025 generates significant economic activity through various channels:

- **Direct Revenue Streams**: Broadcasting rights, ticket sales, merchandise
- **Sponsorship Landscape**: Dominated by fantasy gaming and pan masala companies
- **Sector-wise Impact**: Entertainment, hospitality, and digital economy sectors see significant boosts
- **Income Group Analysis**: All income segments participate in IPL consumption, with varying patterns

### Social Responsibility Analysis

While economically beneficial, certain IPL advertising practices raise concerns:

- **Health Risk Distribution**: High concentration of potentially harmful product advertisements
- **Demographic Vulnerability**: Lower-income groups show greater susceptibility to advertising influence
- **Celebrity Endorsement Effects**: Famous cricketers' endorsements amplify product appeal regardless of health risks
- **Regulatory Considerations**: Current regulations may be insufficient to protect vulnerable populations

## Recommendations

Based on our analysis, we recommend:

1. **For IPL Organizers**:
   - Diversify sponsorship portfolio to reduce dependence on high-risk product categories
   - Implement stronger advertising standards during broadcasts
   - Allocate a percentage of advertising time to public health messages

2. **For Brands**:
   - Develop more socially responsible advertising approaches
   - Consider demographic vulnerability when targeting advertisements
   - Balance profit motives with ethical considerations

3. **For Policymakers**:
   - Review regulations on celebrity endorsements of potentially harmful products
   - Consider stronger warning requirements for high-risk product advertisements
   - Develop education campaigns about responsible consumption

## Conclusion

The IPL 2025 demonstrates the complex interplay between economic benefits and social responsibility. While celebrating its economic contributions, stakeholders must address the potential negative impacts of certain advertising practices, particularly on vulnerable populations.

---

*This report was prepared by Business Basics. Chief Editor: Tony Sharma | Research Lead: Sathish Anantharaj*
"""
    
    # Save the report to a markdown file
    with open('IPL_2025_Impact_Analysis_Report.md', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main function to create and run the dashboard"""
    print("Creating IPL 2025 Impact Analysis Dashboard...")
    
    # Load all data
    dataframes = load_all_data()
    
    # Clean and process data
    clean_dataframes = clean_data(dataframes)
    
    # Analyze economic impact
    economic_data = analyze_economic_impact(clean_dataframes)
    
    # Analyze social impact
    social_data = analyze_social_impact(clean_dataframes)
    
    # Create visualizations
    figures = create_plotly_figures(economic_data, social_data)
    
    # Create the dashboard
    app = create_dashboard(figures)
    
    # Run the dashboard
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get('PORT', 8050))
    
    print(f"Dashboard will be available at http://127.0.0.1:{port}/")
    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
