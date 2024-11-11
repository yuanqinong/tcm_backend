-- Create the customers table
CREATE TABLE customers (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL
);

-- Create the purchase_history table
CREATE TABLE purchase_history (
    id SERIAL PRIMARY KEY,
    customer_uuid UUID NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    product_quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_uuid) REFERENCES customers(uuid)
);

-- Create an index on customer_uuid for faster lookups
CREATE INDEX idx_purchase_history_customer_uuid ON purchase_history(customer_uuid);