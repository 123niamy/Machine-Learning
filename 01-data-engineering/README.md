# Data Engineering

ETL pipelines, data validation, transformation, storage, and governance for production-ready data workflows.

## Projects

### Data Engineering2.py
**Titanic Data Pipeline** - Comprehensive 7-stage ETL pipeline with visualization.

**Pipeline Stages:**
1. **Ingestion**: Load CSV data with error handling
2. **Validation**: Check schema and required columns
3. **Transformation**: Clean missing values, feature engineering (Age groups, Family size)
4. **Storage**: Persist to SQLite database
5. **Processing**: Run SQL aggregations (survival rates by class)
6. **Visualization**: Create survival charts by sex and class
7. **Logging**: Track all operations for governance

**Key Features:**
- Median imputation for missing Age values
- Forward-fill for missing Embarked values
- Feature engineering: Age_Group, Family_Size
- SQLite database with proper context managers
- Matplotlib/Seaborn visualizations saved as PNG
- Comprehensive logging to `titanic_pipeline.log`

**Output:**
- `titanic.db` - SQLite database with cleaned data
- `titanic_survival_by_sex.png` - Survival visualization
- `titanic_survival_by_class.png` - Class-based survival chart
- `titanic_pipeline.log` - Complete execution log

---

### Iris data pipeline1.py
**Iris ETL Pipeline** - 7-stage comprehensive medallion architecture.

**Pipeline Stages:**
1. **Ingestion**: Generate and load Iris CSV data
2. **Preprocessing**: StandardScaler normalization, PCA transformation
3. **Integration**: Join with metadata (source tracking)
4. **Quality Checks**: Validate data ranges and distributions
5. **Governance**: Track data lineage and timestamps
6. **Serving**: Prepare final dataset for consumption
7. **Storage**: Save to multiple formats (CSV, database)

**Key Features:**
- Uses sklearn's Iris dataset
- StandardScaler for feature normalization
- PCA for dimensionality reduction
- Metadata integration for data lineage
- Quality validation with statistical checks
- Governance metadata tracking
- Multiple output formats

**Output:**
- Processed CSV files at each stage
- Quality validation reports
- Governance metadata logs

---

### Data Engineering3.py
**Simple CSV to SQLite Ingestion** - Basic data pipeline pattern.

**What it does:**
- Ingests CSV files into SQLite database
- Command-line argument support for flexibility
- Error handling and validation
- Context manager for safe database operations

**Key Features:**
- Argparse for CLI arguments
- File existence validation
- SQLite integration with pandas
- Clean error messages
- Proper resource cleanup

**Usage:**
```bash
python "Data Engineering3.py" input.csv database.db table_name
```

---

## Data Engineering Concepts Covered

### ETL Fundamentals
- **Extract**: Load data from various sources (CSV, APIs)
- **Transform**: Clean, validate, enrich data
- **Load**: Store in databases or data warehouses

### Data Quality
- Schema validation
- Missing value handling
- Data type checking
- Range validation

### Feature Engineering
- Creating derived features (Age_Group, Family_Size)
- Binning continuous variables
- Aggregating related columns

### Data Governance
- Logging for audit trails
- Metadata tracking
- Data lineage
- Quality metrics

### Storage Patterns
- SQLite for local databases
- Pandas integration
- Context managers for safe operations
- Multiple format exports

---

## Tech Stack

- **Python 3.14**
- **pandas**: Data manipulation
- **SQLite3**: Lightweight database
- **scikit-learn**: Data preprocessing, datasets
- **matplotlib/seaborn**: Visualization
- **logging**: Pipeline monitoring

---

**Best Practices Demonstrated:**
- ✅ Error handling with try/except blocks
- ✅ Logging for observability
- ✅ Context managers for resource management
- ✅ Path-relative file operations
- ✅ Function-based modular design
- ✅ Comprehensive documentation
