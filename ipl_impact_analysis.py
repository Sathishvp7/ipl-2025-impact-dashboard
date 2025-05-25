import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Define the data directory
DATA_DIR = Path('Dataset')

# Load all datasets
def load_data():
    """Load all Excel files from the Dataset directory"""
    print("Loading data files...")
    
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
                import re
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
    
    return dfs

def analyze_economic_impact(dataframes):
    """Analyze the economic impact of IPL"""
    print("Analyzing economic impact...")
    
    contracts_df = dataframes['contracts']
    revenue_df = dataframes['revenue']
    
    # Analyze contract values
    total_sponsorship = contracts_df['amount_in_crores_2025'].sum() if 'amount_in_crores_2025' in contracts_df.columns else 0
    contract_by_type = contracts_df.groupby('contract_type')['amount_in_crores_2025'].sum() if 'contract_type' in contracts_df.columns else pd.Series()
    
    # Analyze revenue impact by company sector
    # Use the cleaned numeric_revenue column instead of the original column
    if 'sector' in revenue_df.columns and 'numeric_revenue' in revenue_df.columns:
        sector_revenue = revenue_df.groupby('sector')['numeric_revenue'].sum()
    else:
        sector_revenue = pd.Series()
    
    return {
        'total_sponsorship': total_sponsorship,
        'contract_by_type': contract_by_type,
        'sector_revenue': sector_revenue
    }

def analyze_social_impact(dataframes):
    """Analyze the social and health implications of IPL advertising"""
    print("Analyzing social impact...")
    
    advertisers_df = dataframes['advertisers']
    
    # Analyze health and social risks
    risk_by_category = advertisers_df.groupby('category')['health_social_risk'].value_counts() if 'category' in advertisers_df.columns and 'health_social_risk' in advertisers_df.columns else pd.Series()
    
    # Analyze celebrity influence on potentially harmful products
    celebrity_impact = advertisers_df.groupby(['celebrity_influence', 'health_social_risk']).size() if 'celebrity_influence' in advertisers_df.columns and 'health_social_risk' in advertisers_df.columns else pd.Series()
    
    return {
        'risk_by_category': risk_by_category,
        'celebrity_impact': celebrity_impact
    }

def create_visualizations(economic_data, social_data, dataframes):
    """Create visualizations for the report"""
    print("Creating visualizations...")
    
    # Create output directory for visualizations
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Economic Impact Visualizations
    if hasattr(economic_data['contract_by_type'], 'plot'):
        plt.figure(figsize=(10, 6))
        ax = economic_data['contract_by_type'].sort_values(ascending=False).plot(kind='bar', color='skyblue')
        
        # Add data labels on top of each bar
        for i, v in enumerate(economic_data['contract_by_type'].sort_values(ascending=False)):
            ax.text(i, v + 5, f'₹{v:.1f} Cr', ha='center', fontweight='bold')
            
        plt.title('IPL 2025 Sponsorship Amount by Contract Type', fontsize=14, fontweight='bold')
        plt.xlabel('Contract Type', fontsize=12)
        plt.ylabel('Amount in Crores (₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'sponsorship_by_contract_type.png', dpi=300)
        plt.close()
    
    if hasattr(economic_data['sector_revenue'], 'plot'):
        plt.figure(figsize=(12, 7))
        ax = economic_data['sector_revenue'].sort_values(ascending=False).plot(
            kind='bar', 
            color=sns.color_palette("viridis", len(economic_data['sector_revenue']))
        )
        
        # Add data labels on top of each bar
        for i, v in enumerate(economic_data['sector_revenue'].sort_values(ascending=False)):
            ax.text(i, v + 5, f'₹{v:.1f} Cr', ha='center', fontweight='bold')
            
        plt.title('Revenue by Sector (Based on First Value in Range)', fontsize=14, fontweight='bold')
        plt.xlabel('Sector', fontsize=12)
        plt.ylabel('Revenue (Crores ₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / 'revenue_by_sector.png', dpi=300)
        plt.close()
    
    # Social Impact Visualizations
    if hasattr(social_data['risk_by_category'], 'unstack'):
        risk_data = social_data['risk_by_category'].unstack().fillna(0)
        plt.figure(figsize=(12, 8))
        ax = risk_data.plot(kind='bar', stacked=True, colormap='viridis')
        
        # Add a table below the chart with the actual values
        risk_table = plt.table(
            cellText=risk_data.values.round(0).astype(int).astype(str),
            rowLabels=risk_data.index,
            colLabels=risk_data.columns,
            loc='bottom', 
            bbox=[0, -0.35, 1, 0.2]
        )
        risk_table.auto_set_font_size(False)
        risk_table.set_fontsize(9)
        
        plt.title('Health and Social Risks by Advertiser Category', fontsize=14, fontweight='bold')
        plt.xlabel('')  # Remove x-label as we have a table below
        plt.ylabel('Count', fontsize=12)
        plt.xticks([])  # Hide x-ticks as we have a table
        plt.legend(title='Risk Level', title_fontsize=12)
        plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin for table
        plt.tight_layout()
        plt.savefig(output_dir / 'health_risks_by_category.png', dpi=300)
        plt.close()
    
    # Create a combined dashboard using Plotly
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("Sponsorship by Contract Type", "Revenue Distribution by Sector", 
                        "Health Risks by Category", "Celebrity Influence on Risky Products")
    )
    
    # Add sponsorship by contract type
    if hasattr(economic_data['contract_by_type'], 'index'):
        sorted_contract = economic_data['contract_by_type'].sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=sorted_contract.index,
                y=sorted_contract.values,
                name="Sponsorship Amount",
                text=[f'₹{v:.1f} Cr' for v in sorted_contract.values],
                textposition='auto',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
    
    # Add revenue by sector
    if hasattr(economic_data['sector_revenue'], 'index'):
        sorted_revenue = economic_data['sector_revenue'].sort_values(ascending=False)
        fig.add_trace(
            go.Pie(
                labels=sorted_revenue.index,
                values=sorted_revenue.values,
                name="Sector Revenue",
                text=[f'₹{v:.1f} Cr' for v in sorted_revenue.values],
                textinfo='label+percent',
                hole=0.4
            ),
            row=1, col=2
        )
    
    # Add risk by category visualization
    if hasattr(social_data['risk_by_category'], 'unstack'):
        risk_data = social_data['risk_by_category'].unstack().fillna(0)
        for i, risk in enumerate(risk_data.columns):
            fig.add_trace(
                go.Bar(
                    x=risk_data.index,
                    y=risk_data[risk],
                    name=risk,
                    text=risk_data[risk].astype(int),
                    textposition='auto'
                ),
                row=2, col=1
            )
    
    # Add celebrity influence visualization
    if hasattr(social_data['celebrity_impact'], 'unstack'):
        celebrity_data = social_data['celebrity_impact'].unstack().fillna(0)
        for i, risk in enumerate(celebrity_data.columns):
            fig.add_trace(
                go.Bar(
                    x=celebrity_data.index,
                    y=celebrity_data[risk],
                    name=risk,
                    text=celebrity_data[risk].astype(int),
                    textposition='auto'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=900, 
        width=1200, 
        title_text="IPL 2025: Economic Benefits vs. Social Implications",
        title_font_size=20,
        legend_title_text="Categories",
        showlegend=True
    )
    
    # Improve layout and formatting
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Amount (₹ Crores)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Save the dashboard
    fig.write_html(output_dir / "ipl_impact_dashboard.html")
    
    print(f"Visualizations saved to {output_dir}")

def demographic_analysis(dataframes):
    """Analyze demographic impact"""
    print("Analyzing demographic impact...")
    
    revenue_df = dataframes['revenue']
    summary_df = dataframes['summary']
    
    # Analyze income group distribution
    income_distribution = revenue_df['income_group'].value_counts() if 'income_group' in revenue_df.columns else pd.Series()
    
    # Analyze age group distribution
    age_distribution = revenue_df['age_group'].value_counts() if 'age_group' in revenue_df.columns else pd.Series()
    
    # Analyze urban population distribution
    urban_distribution = revenue_df['urban_population'].value_counts() if 'urban_population' in revenue_df.columns else pd.Series()
    
    return {
        'income_distribution': income_distribution,
        'age_distribution': age_distribution,
        'urban_distribution': urban_distribution
    }

def generate_report(economic_data, social_data, demographic_data):
    """Generate a comprehensive report with the analysis"""
    print("Generating report...")
    
    report = """
    # IPL 2025: Economic Impact and Social Implications Analysis
    
    ## Executive Summary
    
    This report provides a balanced analysis of IPL 2025's economic contributions alongside potential social and health implications of certain brand partnerships. The analysis is based on data from IPL central contracts, advertisers, and demographic information.
    
    ## Economic Impact
    
    The total sponsorship amount for IPL 2025 is ₹{:,.2f} crores. The distribution of this amount across different contract types shows that {}.
    
    ## Social and Health Implications
    
    The analysis of advertisers reveals that {}. Celebrity endorsements play a significant role in promoting products with varying degrees of health and social risks.
    
    ## Demographic Impact
    
    The IPL advertising primarily targets {} age groups and {} income groups. The urban population distribution shows that {}.
    
    ## Recommendations
    
    1. Balance economic benefits with social responsibility by implementing stricter advertising guidelines.
    2. Encourage more socially responsible brands to partner with IPL.
    3. Develop a rating system for advertisements based on their health and social impact.
    4. Allocate a percentage of advertising revenue to health awareness campaigns.
    5. Introduce time restrictions for potentially harmful product advertisements.
    
    ## Conclusion
    
    IPL 2025 continues to be a significant economic driver, but there's a need for more balanced advertising practices that consider social and health implications.
    """.format(
        economic_data['total_sponsorship'],
        "the highest amount goes to specific contract types" if hasattr(economic_data['contract_by_type'], 'idxmax') else "the distribution data is not available",
        "certain categories pose higher health and social risks" if hasattr(social_data['risk_by_category'], 'unstack') else "the health risk data is not available",
        "specific" if hasattr(demographic_data['age_distribution'], 'idxmax') else "various",
        "specific" if hasattr(demographic_data['income_distribution'], 'idxmax') else "various",
        "urban areas have higher coverage" if hasattr(demographic_data['urban_distribution'], 'idxmax') else "the distribution varies"
    )
    
    # Save the report
    with open('IPL_2025_Impact_Analysis_Report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Report generated: IPL_2025_Impact_Analysis_Report.md")

def main():
    """Main function to run the analysis"""
    # Load data
    dataframes = load_data()
    
    # Clean data
    clean_dfs = clean_data(dataframes)
    
    # Analyze economic impact
    economic_data = analyze_economic_impact(clean_dfs)
    
    # Analyze social impact
    social_data = analyze_social_impact(clean_dfs)
    
    # Analyze demographic impact
    demographic_data = demographic_analysis(clean_dfs)
    
    # Create visualizations
    create_visualizations(economic_data, social_data, clean_dfs)
    
    # Generate report
    generate_report(economic_data, social_data, demographic_data)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
