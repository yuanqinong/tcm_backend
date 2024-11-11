-- Create the products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);

-- Insert dummy products
INSERT INTO products (category, name, price) VALUES
-- Electronics
('Electronics', 'Smartphone', 899.99),
('Electronics', 'Laptop', 1299.99),
('Electronics', 'Wireless Earbuds', 159.99),
('Electronics', 'Smart Watch', 299.99),
('Electronics', 'Tablet', 499.99),
('Electronics', '4K TV', 799.99),
('Electronics', 'Gaming Console', 499.99),

-- Home Appliances
('Home Appliances', 'Refrigerator', 999.99),
('Home Appliances', 'Washing Machine', 599.99),
('Home Appliances', 'Microwave Oven', 149.99),
('Home Appliances', 'Vacuum Cleaner', 199.99),

-- Clothing
('Clothing', 'Men''s Jeans', 59.99),
('Clothing', 'Women''s Dress', 79.99),
('Clothing', 'Winter Jacket', 129.99),
('Clothing', 'Running Shoes', 89.99),

-- Books
('Books', 'Science Fiction Novel', 14.99),
('Books', 'Cookbook', 24.99),
('Books', 'History Book', 19.99),

-- Sports & Outdoors
('Sports & Outdoors', 'Tennis Racket', 79.99),
('Sports & Outdoors', 'Yoga Mat', 29.99),
('Sports & Outdoors', 'Camping Tent', 199.99),

-- Beauty & Personal Care
('Beauty & Personal Care', 'Electric Toothbrush', 69.99),
('Beauty & Personal Care', 'Hair Dryer', 49.99),

-- Toys
('Toys', 'LEGO Set', 59.99),
('Toys', 'Board Game', 39.99);

-- Verify the inserted data
SELECT * FROM products ORDER BY category, name;