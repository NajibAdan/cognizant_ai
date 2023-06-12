# About
A retail grocery store that heavily uses IoT is looking for a way to estimate whether or not a product should be stocked.

# Dataset
- `sales.csv` contains sales data
- `sensor_stock_levels.csv` contains IoT sensor data of estimated stock percentage for a particular product at a particular time
- `sensor_storage_temperature.csv` contains IoT sensor data for the temperature. 

## Data Description for `sales.csv`
- transaction_id = this is a unique ID that is assigned to each transaction
- timestamp = this is the datetime at which the transaction was made
- product_id = this is an ID that is assigned to the product that was sold. Each product has a unique ID
- category = this is the category that the product is contained within
- customer_type = this is the type of customer that made the transaction
- unit_price = the price that 1 unit of this item sells for
- quantity = the number of units sold for this product within this transaction
- total = the total amount payable by the customer
- payment_type = the payment method used by the customer
