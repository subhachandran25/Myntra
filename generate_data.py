import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 5000

PRODUCTS = [
    "Ethnic Kurta", "Denim Jeans", "Casual T-Shirt", "Formal Shirt",
    "Saree", "Lehenga", "Sneakers", "Heels", "Flats", "Sports Shoes",
    "Handbag", "Wallet", "Sunglasses", "Watch", "Jacket",
    "Hoodie", "Palazzo Pants", "Maxi Dress", "Crop Top", "Blazer"
]

CATEGORIES = {
    "Ethnic Kurta": "Ethnic Wear", "Saree": "Ethnic Wear", "Lehenga": "Ethnic Wear",
    "Denim Jeans": "Western Wear", "Casual T-Shirt": "Western Wear", "Formal Shirt": "Western Wear",
    "Jacket": "Western Wear", "Hoodie": "Western Wear", "Palazzo Pants": "Western Wear",
    "Maxi Dress": "Western Wear", "Crop Top": "Western Wear", "Blazer": "Western Wear",
    "Sneakers": "Footwear", "Heels": "Footwear", "Flats": "Footwear", "Sports Shoes": "Footwear",
    "Handbag": "Accessories", "Wallet": "Accessories", "Sunglasses": "Accessories", "Watch": "Accessories"
}

RETURN_REASONS = [
    "Size Mismatch", "Quality Issue", "Wrong Item Delivered",
    "Color Difference", "Damaged Product", "Changed Mind",
    "Better Price Found", "Fit Issue", "Fabric Issue", "Late Delivery"
]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]

SEGMENTS = ["Champions", "Loyal Customers", "At Risk", "New Customers", "Lost Customers"]

base_date = datetime(2023, 1, 1)

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, 1001)]
customer_pool = np.random.choice(customer_ids, N)

purchase_dates = [base_date + timedelta(days=np.random.randint(0, 730)) for _ in range(N)]

products = np.random.choice(PRODUCTS, N)
categories = [CATEGORIES[p] for p in products]

quantities = np.random.choice([1, 2, 3, 4, 5], N, p=[0.55, 0.25, 0.1, 0.05, 0.05])

base_prices = {
    "Ethnic Kurta": 899, "Denim Jeans": 1499, "Casual T-Shirt": 499,
    "Formal Shirt": 1299, "Saree": 2499, "Lehenga": 4999,
    "Sneakers": 1999, "Heels": 2499, "Flats": 1299, "Sports Shoes": 2999,
    "Handbag": 3499, "Wallet": 999, "Sunglasses": 1499, "Watch": 4999,
    "Jacket": 2999, "Hoodie": 1799, "Palazzo Pants": 899, "Maxi Dress": 1999,
    "Crop Top": 699, "Blazer": 3499
}

discount_pct = np.random.choice([0, 10, 20, 30, 40, 50, 60, 70], N,
                                  p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])

amounts = []
for i in range(N):
    price = base_prices[products[i]]
    disc = discount_pct[i] / 100
    amounts.append(round(price * quantities[i] * (1 - disc), 2))

# Return probability — influenced by discount, category, rating
return_prob_base = {
    "Ethnic Wear": 0.38, "Western Wear": 0.35, "Footwear": 0.40,
    "Accessories": 0.22
}

ratings = np.random.choice([1, 2, 3, 4, 5], N, p=[0.08, 0.12, 0.20, 0.35, 0.25])

return_requested = []
for i in range(N):
    cat = categories[i]
    base_p = return_prob_base[cat]
    # Higher discount -> more returns (impulse)
    disc_factor = 1 + (discount_pct[i] / 100) * 0.5
    # Lower rating -> more returns
    rating_factor = 1 + (3 - ratings[i]) * 0.1
    p = min(base_p * disc_factor * rating_factor, 0.85)
    return_requested.append(np.random.binomial(1, p))

return_requested = np.array(return_requested)

return_reasons = []
for rr in return_requested:
    if rr == 1:
        return_reasons.append(np.random.choice(RETURN_REASONS))
    else:
        return_reasons.append(None)

date_of_return = []
for i, rr in enumerate(return_requested):
    if rr == 1:
        days_to_return = np.random.randint(1, 30)
        date_of_return.append(purchase_dates[i] + timedelta(days=days_to_return))
    else:
        date_of_return.append(None)

quantity_returned = []
for i, rr in enumerate(return_requested):
    if rr == 1:
        qty_ret = np.random.randint(1, int(quantities[i]) + 1)
        quantity_returned.append(qty_ret)
    else:
        quantity_returned.append(0)

amount_returned = []
for i, rr in enumerate(return_requested):
    if rr == 1:
        per_unit = amounts[i] / quantities[i]
        amount_returned.append(round(per_unit * quantity_returned[i], 2))
    else:
        amount_returned.append(0.0)

cities = np.random.choice(CITIES, N)
is_sale_period = []
SALE_DATES = [
    (datetime(2023, 6, 1), datetime(2023, 6, 10)),
    (datetime(2023, 12, 15), datetime(2023, 12, 25)),
    (datetime(2024, 6, 1), datetime(2024, 6, 10)),
]
for pd_date in purchase_dates:
    in_sale = any(s <= pd_date <= e for s, e in SALE_DATES)
    is_sale_period.append(int(in_sale))

# RFM + Customer Segment
customer_rfm = {}
for cust in customer_ids:
    recency = np.random.randint(1, 365)
    frequency = np.random.randint(1, 20)
    monetary = np.random.randint(500, 50000)
    r_score = 5 - min(4, recency // 73)
    f_score = min(5, frequency // 4 + 1)
    m_score = min(5, monetary // 10000 + 1)
    rfm = r_score + f_score + m_score
    if rfm >= 12:
        seg = "Champions"
    elif rfm >= 9:
        seg = "Loyal Customers"
    elif rfm >= 6:
        seg = "At Risk"
    elif rfm >= 4:
        seg = "New Customers"
    else:
        seg = "Lost Customers"
    customer_rfm[cust] = {
        "recency": recency, "frequency": frequency,
        "monetary": monetary, "rfm_score": rfm, "segment": seg
    }

segments = [customer_rfm[c]["segment"] for c in customer_pool]
rfm_scores = [customer_rfm[c]["rfm_score"] for c in customer_pool]
recency_days = [customer_rfm[c]["recency"] for c in customer_pool]
frequency_vals = [customer_rfm[c]["frequency"] for c in customer_pool]
monetary_vals = [customer_rfm[c]["monetary"] for c in customer_pool]

df = pd.DataFrame({
    "customer_id": customer_pool,
    "purchase_date": purchase_dates,
    "product": products,
    "category": categories,
    "quantity": quantities,
    "amount": amounts,
    "discount_pct": discount_pct,
    "is_sale_period": is_sale_period,
    "return_requested": return_requested,
    "return_reason": return_reasons,
    "date_of_return": date_of_return,
    "quantity_returned": quantity_returned,
    "amount_returned": amount_returned,
    "product_rating": ratings,
    "city": cities,
    "customer_segment": segments,
    "rfm_score": rfm_scores,
    "recency_days": recency_days,
    "purchase_frequency": frequency_vals,
    "total_spend": monetary_vals,
})

df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["date_of_return"] = pd.to_datetime(df["date_of_return"])
df["year_month"] = df["purchase_date"].dt.to_period("M").astype(str)

df.to_csv("myntra_data.csv", index=False)
print(f"Dataset created: {len(df)} rows")
print(df.dtypes)
print(df["return_requested"].value_counts())
