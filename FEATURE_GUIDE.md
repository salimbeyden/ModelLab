# Quick Start Guide - New Features

## 🎯 Feature 1: Custom Train/Test Split Ratio

### Where to Find It
Navigate to: **New Training Run** page

### How to Use
1. Select your dataset and target variable
2. Choose your task type (regression/classification)
3. **NEW: Train/Test Split Slider**
   - Located below the Task & Model selection
   - Drag the slider to adjust the split ratio
   - Range: 10% to 50% test data
   - Default: 20% test data (80% train / 20% test)
   - Real-time visual feedback shows percentages

### Visual Example
```
Train/Test Split Ratio
[============================|====] 
Train: 80% | Test: 20%

Adjust the proportion of data used for testing (validation)
```

### When to Adjust
- **Small datasets (<1000 rows)**: Use 90/10 or 85/15 split
- **Medium datasets (1000-10000 rows)**: Use 80/20 split (default)
- **Large datasets (>10000 rows)**: Use 70/30 or 75/25 split

---

## 🏆 Feature 2: Models Management Page

### Where to Find It
- **Navigation Bar**: Click "Models" button (between Datasets and New Training Run)
- **Home Page**: Click "View Models" card
- **Direct URL**: http://localhost:3000/models

### What You'll See

#### Filter & Sort Controls
```
[All (5)] [Regression (3)] [Classification (2)]     Sort by: [Date (Newest) ▼]
```

#### Model Cards Grid
Each model card displays:
- 🤖 Model type icon (EBM or mgcv)
- 📅 Training date
- 🏷️ Task type badge (regression/classification)
- 🎯 Target variable
- 📊 Primary metric (R² or Accuracy)
- 📁 Number of artifacts
- 👆 Click anywhere to view full details

#### Stats Dashboard (Bottom)
```
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ Total Models   │ Regression     │ Classification │ Avg Performance│
│      12        │       7        │       5        │     0.856      │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

### Features

#### 1. Filtering
- **All**: Shows all completed models
- **Regression**: Shows only regression models
- **Classification**: Shows only classification models

#### 2. Sorting
- **Date (Newest)**: Most recent models first (default)
- **Performance**: Best performing models first
  - Regression: Sorted by R² score
  - Classification: Sorted by accuracy

#### 3. Quick Actions
- Click any model card → Navigate to detailed run results
- View metrics, artifacts, and visualizations
- Access model explanations

---

## 📊 Updated Navigation

Your navigation bar now has 4 buttons:

```
┌──────┬──────────┬────────┬─────────────────────┐
│ Home │ Datasets │ Models │ New Training Run    │
└──────┴──────────┴────────┴─────────────────────┘
```

1. **Home** - Dashboard overview
2. **Datasets** - Manage your data
3. **Models** - Browse trained models (NEW!)
4. **New Training Run** - Start training

---

## 💡 Usage Examples

### Example 1: Train with Different Splits
```
Experiment 1: Default split (80/20)
- Dataset: housing_data.csv
- Model: EBM
- Split: 80% train / 20% test
- Result: R² = 0.85

Experiment 2: More training data (90/10)
- Dataset: housing_data.csv (same)
- Model: EBM (same params)
- Split: 90% train / 10% test
- Result: R² = 0.87 (improved!)
```

### Example 2: Compare Models
1. Train multiple models with different configurations
2. Go to Models page
3. Sort by "Performance"
4. See your best model at the top
5. Click to view details and download artifacts

### Example 3: Find Specific Model
1. Go to Models page
2. Filter by task type (e.g., "Regression")
3. Look for specific target variable
4. Click to reuse or analyze further

---

## 🎨 Visual Design

### Train/Test Slider
- **Color**: Blue for train, Green for test
- **Interactive**: Smooth drag experience
- **Feedback**: Instant percentage updates

### Models Page
- **Cards**: Clean, professional design
- **Colors**: 
  - Blue theme for EBM models
  - Purple theme for mgcv models
  - Green badges for regression
  - Yellow badges for classification
- **Hover Effects**: Cards lift on hover
- **Empty State**: Helpful message when no models exist

---

## 🚀 Next Steps

After implementing these features, you can:

1. **Train multiple models** with different split ratios
2. **Compare performance** using the Models page
3. **Find your best model** by sorting by performance
4. **Reuse models** for future analysis
5. **Track your ML experiments** systematically

---

## 📝 Tips

### Best Practices
- ✅ Start with default 80/20 split
- ✅ Adjust based on dataset size
- ✅ Train multiple models for comparison
- ✅ Use Models page to track experiments
- ✅ Document why you chose specific split ratios

### Common Patterns
- **Small dataset** → More training data (90/10)
- **Imbalanced classes** → Consider stratified split (future feature)
- **Time series** → Use temporal split (future feature)
- **A/B testing** → Compare models with same split ratio

---

## 🆘 Troubleshooting

### Models page is empty
- Make sure you have completed at least one training run
- Check that runs reached "completed" status
- Refresh the page

### Split ratio not changing
- Make sure slider is visible in New Training Run page
- Try clicking and dragging the slider
- Check browser console for errors

### Model cards not showing metrics
- Some older runs may not have all metrics
- Retrain the model to populate all fields
- Check run status (must be "completed")

---

## 🎉 Summary

You now have:
1. ✅ Flexible train/test split ratio control
2. ✅ Centralized model management system
3. ✅ Performance-based model comparison
4. ✅ Better experiment tracking
5. ✅ Improved ML workflow

Happy modeling! 🚀
