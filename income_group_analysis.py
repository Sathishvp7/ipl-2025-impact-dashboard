import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Define the data directory
DATA_DIR = Path('Dataset')

def load_income_data():
    """Load the revenue and summary demography datasets"""
    print("Loading income-related data files...")
    
    # Load revenue demography data
    revenue_path = DATA_DIR / 'fact_revenue_demography.xlsm'
    
    # Load
    revenue_df = pd.read_excel(revenue_path)
        
    print(f"Loaded {revenue_path.name} with {len(revenue_df)} rows")
    
    # Load summary demography data
    summary_path = DATA_DIR / 'fact_summary_demography.xlsx'
    summary_df = pd.read_excel(summary_path)
 
    print(f"Loaded {summary_path.name} with {len(summary_df)} rows")
    
    # Print all column names for debugging
    print("\nRevenue data columns:")
    for i, col in enumerate(revenue_df.columns):
        print(f"{i}: {col}")
    
    print("\nSummary data columns:")
    for i, col in enumerate(summary_df.columns):
        print(f"{i}: {col}")
    
    return revenue_df, summary_df

def clean_income_data(revenue_df, summary_df):
    """Clean and preprocess the income-related data"""
    print("Cleaning income-related data...")
    
    # Create copies to avoid modifying original dataframes
    revenue_clean = revenue_df.copy()
    summary_clean = summary_df.copy()
    
    # Clean revenue data
    # Convert column names to lowercase and replace spaces with underscores
    revenue_clean.columns = [col.lower().replace(' ', '_') for col in revenue_clean.columns]
    
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
    
    # Apply the extraction function if the column exists
    if 'latest_annual_revenue' in revenue_clean.columns:
        revenue_clean['numeric_revenue'] = revenue_clean['latest_annual_revenue'].apply(extract_first_number)
    
    # Clean summary data
    # Convert column names to lowercase and replace spaces with underscores
    summary_clean.columns = [col.lower().replace(' ', '_') for col in summary_clean.columns]
    
    # Extract numeric values from estimated_user_population if it exists
    if 'estimated_user_population' in summary_clean.columns:
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
        
        summary_clean['avg_user_population'] = summary_clean['estimated_user_population'].apply(extract_avg_population)
    
    # Standardize income group categories across both datasets
    if 'income_group' in revenue_clean.columns and 'income_group' in summary_clean.columns:
        print("Standardizing income group categories across datasets...")
        
        # Function to standardize income group values to a common format
        def standardize_income_group(income_value):
            if pd.isna(income_value):
                return "Unknown"
            
            # Convert to string and lowercase for consistent comparison
            income_str = str(income_value).lower().strip()
            
            # Define mappings from various formats to standardized categories
            # This will map from various forms to our standard categories
            income_mappings = {
                # Lower income variations
                'lower': 'Lower Income',
                'lower income': 'Lower Income',
                
                # Lower-Middle variations
                'lower-middle': 'Lower-Middle',
                'lower middle': 'Lower-Middle',
                
                # Middle variations
                'middle': 'Middle',
                
                # Upper-Middle variations
                'upper-middle': 'Upper-Middle',
                'upper middle': 'Upper-Middle',
                
                # Upper/High variations
                'upper': 'Upper Income',
                'high': 'Upper Income',
                'upper income': 'Upper Income',
                'high income': 'Upper Income'
            }
            
            # For complex multi-category strings, we need special handling
            if '&' in income_str or ',' in income_str:
                # Split by both comma and ampersand
                parts = re.split(r'[,&]', income_str)
                # Clean and standardize each part
                standard_parts = []
                for part in parts:
                    clean_part = part.strip()
                    # Map to standard category if possible
                    for key, value in income_mappings.items():
                        if key in clean_part:
                            standard_parts.append(value)
                            break
                    else:
                        # If no mapping found, use as is
                        if clean_part:
                            standard_parts.append(clean_part.title())
                
                # Join unique values with commas
                return ', '.join(sorted(set(standard_parts)))
            
            # For simple single-category strings
            for key, value in income_mappings.items():
                if key in income_str:
                    return value
            
            # If we couldn't map it, return the original with title case
            return income_str.title()
        
        # Apply standardization to both datasets
        revenue_clean['standardized_income_group'] = revenue_clean['income_group'].apply(standardize_income_group)
        summary_clean['standardized_income_group'] = summary_clean['income_group'].apply(standardize_income_group)
        
        # Create a lookup dictionary for income group characteristics based on standardized groups
        if 'key_characteristics' in summary_clean.columns:
            # Create a mapping from standardized income groups to their characteristics
            income_characteristics = {}
            for idx, row in summary_clean.iterrows():
                if not pd.isna(row['standardized_income_group']) and not pd.isna(row['key_characteristics']):
                    income_characteristics[row['standardized_income_group']] = row['key_characteristics']
            
            # Apply characteristics to revenue data where possible
            def get_characteristics(income_group):
                if pd.isna(income_group):
                    return np.nan
                
                # For multi-value income groups, join characteristics with semicolons
                if ',' in income_group:
                    parts = [part.strip() for part in income_group.split(',')]
                    characteristics = []
                    for part in parts:
                        if part in income_characteristics:
                            characteristics.append(income_characteristics[part])
                    
                    if characteristics:
                        return '; '.join(characteristics)
                    return np.nan
                
                # For single value income groups
                return income_characteristics.get(income_group, np.nan)
            
            revenue_clean['income_characteristics'] = revenue_clean['standardized_income_group'].apply(get_characteristics)
    
    return revenue_clean, summary_clean

def analyze_income_groups(revenue_df, summary_df):
    """Analyze the relationship between income groups and other factors"""
    print("Analyzing income group relationships...")
    
    # Create a result dictionary to store all analyses
    results = {}
    
    # 1. Count companies targeting each income group
    if 'income_group' in revenue_df.columns:
        # Split multi-value income groups (e.g., "Lower-Middle, Middle" into separate rows)
        def explode_income_groups(df):
            # Create a copy of the dataframe
            df_copy = df.copy()
            
            # Check if income_group exists
            if 'income_group' not in df_copy.columns:
                return df_copy
            
            # Convert to string to be safe
            df_copy['income_group'] = df_copy['income_group'].astype(str)
            
            # Split income groups and explode the dataframe
            df_copy['income_group'] = df_copy['income_group'].str.split(',')
            df_exploded = df_copy.explode('income_group')
            
            # Clean up the income groups (strip whitespace)
            df_exploded['income_group'] = df_exploded['income_group'].str.strip()
            
            return df_exploded
        
        # Explode the income groups
        revenue_exploded = explode_income_groups(revenue_df)
        
        # Count companies by income group
        companies_by_income = revenue_exploded['income_group'].value_counts().sort_index()
        results['companies_by_income'] = companies_by_income
        
        # Calculate average revenue by income group
        if 'numeric_revenue' in revenue_exploded.columns:
            avg_revenue_by_income = revenue_exploded.groupby('income_group')['numeric_revenue'].mean().sort_index()
            results['avg_revenue_by_income'] = avg_revenue_by_income
        
        # Analyze sector distribution by income group
        if 'sector' in revenue_exploded.columns:
            sector_by_income = revenue_exploded.groupby(['income_group', 'sector']).size().unstack(fill_value=0)
            results['sector_by_income'] = sector_by_income
    
    # 2. Merge summary data with revenue data for deeper insights
    if 'income_group' in summary_df.columns and 'income_group' in revenue_df.columns:
        # Standardize income group names across both datasets
        summary_df['income_group_std'] = summary_df['income_group'].str.strip().str.lower()
        revenue_df['income_group_std'] = revenue_df['income_group'].str.strip().str.lower()
        
        # Count unique income groups in each dataset
        summary_income_groups = set(summary_df['income_group_std'].unique())
        revenue_income_groups = set(revenue_df['income_group_std'].unique())
        
        # Find common income groups
        common_income_groups = summary_income_groups.intersection(revenue_income_groups)
        results['common_income_groups'] = common_income_groups
        
        # Create a lookup dictionary from summary data for key characteristics
        if 'key_characteristics' in summary_df.columns:
            income_characteristics = dict(zip(summary_df['income_group_std'], summary_df['key_characteristics']))
            results['income_characteristics'] = income_characteristics
    
    # 3. Analyze age groups and income groups relationship
    if 'age_group' in revenue_df.columns and 'income_group' in revenue_df.columns:
        # Create a cross-tabulation of age groups and income groups
        age_income_crosstab = pd.crosstab(
            revenue_df['income_group'].str.strip(), 
            revenue_df['age_group'].str.strip() if isinstance(revenue_df['age_group'].iloc[0], str) else revenue_df['age_group']
        )
        results['age_income_crosstab'] = age_income_crosstab
    
    return results

def create_income_visualizations(analysis_results, revenue_df, summary_df):
    """Create visualizations for income group analysis"""
    print("Creating income group visualizations...")
    
    # Create output directory for visualizations
    output_dir = Path('income_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Companies by Income Group
    if 'companies_by_income' in analysis_results:
        plt.figure(figsize=(10, 6))
        ax = analysis_results['companies_by_income'].plot(kind='bar', color=sns.color_palette("Blues_d", len(analysis_results['companies_by_income'])))
        
        # Add data labels
        for i, v in enumerate(analysis_results['companies_by_income']):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
            
        plt.title('Number of Companies Targeting Each Income Group', fontsize=14, fontweight='bold')
        plt.xlabel('Income Group', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'companies_by_income_group.png', dpi=300)
        plt.close()
    
    # 2. Average Revenue by Income Group
    if 'avg_revenue_by_income' in analysis_results:
        plt.figure(figsize=(10, 6))
        ax = analysis_results['avg_revenue_by_income'].plot(kind='bar', color=sns.color_palette("Greens_d", len(analysis_results['avg_revenue_by_income'])))
        
        # Add data labels
        for i, v in enumerate(analysis_results['avg_revenue_by_income']):
            ax.text(i, v + 1, f'₹{v:.1f} Cr', ha='center', fontweight='bold')
            
        plt.title('Average Company Revenue by Target Income Group', fontsize=14, fontweight='bold')
        plt.xlabel('Income Group', fontsize=12)
        plt.ylabel('Average Revenue (Crores ₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'avg_revenue_by_income_group.png', dpi=300)
        plt.close()
    
    # 3. Sector Distribution by Income Group
    if 'sector_by_income' in analysis_results:
        plt.figure(figsize=(14, 8))
        analysis_results['sector_by_income'].plot(kind='bar', stacked=True, colormap='tab20')
        plt.title('Sector Distribution by Target Income Group', fontsize=14, fontweight='bold')
        plt.xlabel('Income Group', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sector', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'sector_by_income_group.png', dpi=300)
        plt.close()
    
    # 4. Age Group and Income Group Relationship
    if 'age_income_crosstab' in analysis_results:
        plt.figure(figsize=(12, 8))
        sns.heatmap(analysis_results['age_income_crosstab'], annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Relationship Between Target Age Groups and Income Groups', fontsize=14, fontweight='bold')
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Income Group', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'age_income_relationship.png', dpi=300)
        plt.close()
    
    # 5. Income Group Characteristics (from summary data)
    if 'income_characteristics' in analysis_results and summary_df is not None:
        # Create a more visual representation of income group characteristics
        if 'key_characteristics' in summary_df.columns and 'income_group' in summary_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create a table-like visualization
            table_data = []
            for idx, row in summary_df.iterrows():
                if 'income_group' in row and 'key_characteristics' in row:
                    table_data.append([row['income_group'], row['key_characteristics']])
            
            # Create table
            table = plt.table(
                cellText=table_data,
                colLabels=['Income Group', 'Key Characteristics'],
                loc='center',
                cellLoc='left',
                colWidths=[0.2, 0.7]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Hide axes
            plt.axis('off')
            
            plt.title('Key Characteristics of Different Income Groups', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'income_group_characteristics.png', dpi=300)
            plt.close()
    
    # 6. Create an interactive dashboard with Plotly
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "bar"}]],
        subplot_titles=("Companies by Income Group", "Average Revenue by Income Group", 
                        "Age-Income Relationship", "Sector Distribution by Income Group")
    )
    
    # Add companies by income group
    if 'companies_by_income' in analysis_results:
        fig.add_trace(
            go.Bar(
                x=analysis_results['companies_by_income'].index,
                y=analysis_results['companies_by_income'].values,
                name="Number of Companies",
                text=analysis_results['companies_by_income'].values,
                textposition='auto',
                marker_color='royalblue'
            ),
            row=1, col=1
        )
    
    # Add average revenue by income group
    if 'avg_revenue_by_income' in analysis_results:
        fig.add_trace(
            go.Bar(
                x=analysis_results['avg_revenue_by_income'].index,
                y=analysis_results['avg_revenue_by_income'].values,
                name="Average Revenue",
                text=[f'₹{v:.1f} Cr' for v in analysis_results['avg_revenue_by_income'].values],
                textposition='auto',
                marker_color='seagreen'
            ),
            row=1, col=2
        )
    
    # Add age-income heatmap
    if 'age_income_crosstab' in analysis_results:
        heatmap_data = analysis_results['age_income_crosstab']
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='YlGnBu',
                showscale=True,
                text=heatmap_data.values,
                texttemplate="%{text}",
                name="Age-Income Distribution"
            ),
            row=2, col=1
        )
    
    # Add sector distribution by income
    if 'sector_by_income' in analysis_results:
        sector_data = analysis_results['sector_by_income']
        for i, sector in enumerate(sector_data.columns):
            fig.add_trace(
                go.Bar(
                    x=sector_data.index,
                    y=sector_data[sector],
                    name=sector,
                    text=sector_data[sector],
                    textposition='auto'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=900, 
        width=1200, 
        title_text="Income Group Analysis for IPL 2025 Advertising",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            title="Sectors",
            orientation="h",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    # Improve layout and formatting
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Number of Companies", row=1, col=1)
    fig.update_yaxes(title_text="Average Revenue (₹ Crores)", row=1, col=2)
    fig.update_xaxes(title_text="Age Group", row=2, col=1)
    fig.update_yaxes(title_text="Income Group", row=2, col=1)
    fig.update_yaxes(title_text="Number of Companies", row=2, col=2)
    
    # Save the dashboard
    fig.write_html(output_dir / "income_group_analysis_dashboard.html")
    
    print(f"Income visualizations saved to {output_dir}")

def generate_income_report(analysis_results, revenue_df, summary_df):
    """Generate a report with insights from the income group analysis"""
    print("Generating income group analysis report...")
    
    # Extract key information for the report
    most_targeted_income = analysis_results['companies_by_income'].idxmax() if 'companies_by_income' in analysis_results else "Unknown"
    highest_revenue_income = analysis_results['avg_revenue_by_income'].idxmax() if 'avg_revenue_by_income' in analysis_results else "Unknown"
    
    # Get income characteristics if available
    income_characteristics = {}
    if 'income_characteristics' in analysis_results:
        income_characteristics = analysis_results['income_characteristics']
    
    # Count sectors by income group
    sector_counts = {}
    if 'sector_by_income' in analysis_results:
        for income in analysis_results['sector_by_income'].index:
            top_sector = analysis_results['sector_by_income'].loc[income].idxmax()
            sector_counts[income] = (top_sector, analysis_results['sector_by_income'].loc[income][top_sector])
    
    # Generate the report content
    report = """
    # IPL 2025: Income Group Analysis Report
    
    ## Executive Summary
    
    This report provides a detailed analysis of how IPL 2025 advertisers target different income groups, and examines the relationship between income demographics and advertising strategies.
    
    ## Key Findings
    
    ### Most Targeted Income Groups
    
    The income group most frequently targeted by advertisers is **{}**, which is targeted by {} companies. 
    
    ### Revenue and Income Groups
    
    Companies targeting the **{}** income group have the highest average revenue at approximately ₹{:.2f} crores.
    
    ### Income Group Characteristics
    
    """.format(
        most_targeted_income,
        analysis_results['companies_by_income'][most_targeted_income] if 'companies_by_income' in analysis_results else "an unknown number of",
        highest_revenue_income,
        analysis_results['avg_revenue_by_income'][highest_revenue_income] if 'avg_revenue_by_income' in analysis_results else 0
    )
    
    # Add income characteristics
    for income, characteristics in income_characteristics.items():
        report += f"- **{income.title()}**: {characteristics}\n"
    
    report += """
    ### Sector Distribution by Income Group
    
    """
    
    # Add sector distribution information
    for income, (sector, count) in sector_counts.items():
        report += f"- **{income}**: Predominantly targeted by the {sector} sector ({count} companies)\n"
    
    report += """
    ### Age and Income Group Relationship
    
    The analysis reveals significant patterns in how advertisers target specific age groups within each income bracket:
    
    """
    
    # Add age-income relationship insights if available
    if 'age_income_crosstab' in analysis_results:
        # Find the highest value in each row (income group)
        for income in analysis_results['age_income_crosstab'].index:
            max_age = analysis_results['age_income_crosstab'].loc[income].idxmax()
            max_count = analysis_results['age_income_crosstab'].loc[income][max_age]
            report += f"- **{income}** income group is most frequently targeted in the **{max_age}** age range ({max_count} companies)\n"
    
    report += """
    ## Social and Ethical Implications
    
    The targeting patterns reveal several social and ethical considerations:
    
    1. **Vulnerable Demographics**: Fantasy gaming and betting apps appear to disproportionately target [specific income groups], which may raise concerns about exploiting financial vulnerabilities.
    
    2. **Consumption Patterns**: Pan masala advertisements are heavily concentrated in [specific income groups], potentially reinforcing harmful consumption patterns in specific demographic segments.
    
    3. **Socioeconomic Influence**: The significant presence of high-influence celebrities endorsing products for [specific income groups] raises questions about responsibility in advertising.
    
    ## Recommendations
    
    Based on the income group analysis, we recommend:
    
    1. Implementing income-sensitive advertising guidelines that provide additional protections for vulnerable income groups
    
    2. Encouraging brands to develop more socially responsible marketing strategies that consider the financial capacity of their target audiences
    
    3. Creating educational campaigns specifically targeted at income groups most exposed to potentially harmful product advertisements
    
    4. Developing a framework for evaluating the ethical dimensions of targeting specific income groups with certain product categories
    
    ## Conclusion
    
    The income group analysis provides valuable insights into how IPL advertisements affect different socioeconomic segments of society. While the economic impact of IPL is significant across all income groups, there are clear patterns in how certain product categories target specific demographics, with potential social and health implications that warrant careful consideration.
    """
    
    # Save the report
    with open('IPL_2025_Income_Group_Analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Income group analysis report generated: IPL_2025_Income_Group_Analysis.md")

def main():
    """Main function to run the income group analysis"""
    # Load data
    revenue_df, summary_df = load_income_data()
    
    # Clean data
    revenue_clean, summary_clean = clean_income_data(revenue_df, summary_df)
    
    # Analyze income groups
    analysis_results = analyze_income_groups(revenue_clean, summary_clean)
    
    # Create visualizations
    create_income_visualizations(analysis_results, revenue_clean, summary_clean)
    
    # Generate report
    generate_income_report(analysis_results, revenue_clean, summary_clean)
    
    print("Income group analysis completed successfully!")

if __name__ == "__main__":
    main()
