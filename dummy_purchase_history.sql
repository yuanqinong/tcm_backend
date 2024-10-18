-- Insert John Doe into the customers table
INSERT INTO customers (username) VALUES ('john_doe')
ON CONFLICT (username) DO NOTHING;

-- Get John Doe's UUID
WITH john_doe_uuid AS (
    SELECT uuid FROM customers WHERE username = 'john_doe'
)

-- Insert 5 electronic products into purchase_history with specific dates
INSERT INTO purchase_history (customer_uuid, product_category, product_name, product_quantity, price, purchase_date)
VALUES
    ((SELECT uuid FROM john_doe_uuid), 'Electronics', 'Smartphone', 1, 899.99, '2024-10-15 10:30:00'),
    ((SELECT uuid FROM john_doe_uuid), 'Electronics', 'Laptop', 1, 1299.99, '2024-10-14 15:45:00'),
    ((SELECT uuid FROM john_doe_uuid), 'Electronics', 'Wireless Earbuds', 1, 159.99, '2024-10-13 09:15:00'),
    ((SELECT uuid FROM john_doe_uuid), 'Electronics', 'Smart Watch', 1, 299.99, '2024-10-12 14:20:00'),
    ((SELECT uuid FROM john_doe_uuid), 'Electronics', 'Tablet', 1, 499.99, '2024-10-11 11:05:00');

-- Verify the inserted data
SELECT c.username, ph.product_name, ph.product_quantity, ph.price, ph.purchase_date
FROM customers c
JOIN purchase_history ph ON c.uuid = ph.customer_uuid
WHERE c.username = 'john_doe'
ORDER BY ph.purchase_date DESC;