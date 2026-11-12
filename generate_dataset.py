"""
Predicting Product Ratings Using Customer Reviews and Product Features
Dataset Generation Script: generate_dataset.py

Generates a realistic synthetic dataset containing exactly 50,000 product reviews
with strict zero-leakage customer review history tracking.
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

TOTAL_RECORDS = 50000
NUM_CUSTOMERS = 7500  # At least 5,000 unique customers
NUM_PRODUCTS = 600    # 100 products per category

# 1. Define Product Categories and Realistic Price Ranges
CATEGORIES = [
    "Electronics",
    "Clothing",
    "Books",
    "Home & Kitchen",
    "Beauty",
    "Sports"
]

PRICE_RANGES = {
    "Electronics": (19.99, 1199.99),
    "Clothing": (9.99, 249.99),
    "Books": (4.99, 59.99),
    "Home & Kitchen": (14.99, 399.99),
    "Beauty": (6.99, 149.99),
    "Sports": (11.99, 349.99)
}

# 2. Review Text Templates Correlated with Rating and Sentiment
REVIEW_TEMPLATES = {
    5: [
        "Outstanding quality! Exceeded all my expectations. The build is sturdy and it works like a charm. Highly recommend to everyone.",
        "Absolutely fantastic product! Premium feel, excellent performance, and arrived in perfect condition. Worth every single penny.",
        "Blown away by how good this is! Five stars without hesitation. Very fast delivery and top tier customer support.",
        "The best purchase I made this entire year. Extremely reliable, durable, and remarkably easy to use. 10/10!",
        "Superb craftsmanship and great attention to detail. Works flawlessly right out of the box. Couldn't be happier!",
        "Incredible value and stellar quality! Has made my daily routine so much better. Highly satisfied with this order.",
        "Exceptional product! Truly impressed with the durability and elegant design. Will definitely buy again from this brand.",
        "Brilliant design, sturdy construction, and does exactly what is promised. An absolute must buy!"
    ],
    4: [
        "Solid product overall. Works very well and provides great value for the price. Minor packaging scuff but the product is great.",
        "Very pleased with this purchase. Good quality materials and dependable performance. A solid four-star choice.",
        "Performs as advertised. Good craftsmanship and simple setup. Only a slight learning curve, but definitely worth it.",
        "Really nice product for this price point. Does the job effectively with only very minor shortcomings.",
        "Great addition to my collection. High quality finish and works reliably. Would recommend to friends.",
        "Pretty impressed with the performance. Shipping was quick and the item matches the description accurately.",
        "Reliable and well designed. Satisfied with the purchase, just minor room for slight aesthetic improvements."
    ],
    3: [
        "Average quality for the price. Works okay but nothing extraordinary. Neither overly impressed nor too disappointed.",
        "Decent product with acceptable build quality. It gets the job done but could definitely use some design improvements.",
        "Mediocre experience. It is fine for basic everyday use, but do not expect top tier or premium performance.",
        "Fair value. It functions adequately, though the material feels slightly more lightweight than expected.",
        "Okay product. Does what it needs to do, but nothing standout. Moderate quality and standard shipping speed.",
        "Middle of the road. Has a few nice aspects but also some minor quirks that keep it from being great."
    ],
    2: [
        "Quite disappointed with this item. Feels cheaply made and did not hold up well under normal use. Not worth the price.",
        "Below average performance. Started malfunctioning within two weeks of light use. Customer service was unhelpful.",
        "Poor quality control. The fit and finish are lacking and it does not work as claimed in the product description.",
        "Would not recommend this. Subpar materials and underwhelming experience overall. Expected much better.",
        "Frustrating experience. Lacks durability and stopped working properly soon after arrival. Regret buying this.",
        "Fell short of expectations. Build quality is flimsy and customer support was unresponsive."
    ],
    1: [
        "Terrible product! Arrived defective and stopped working on the first day. Complete waste of money.",
        "Extremely poor quality. Fell apart almost immediately. Horrible purchase, stay away!",
        "Awful experience! Does not match the description at all. Do not waste your hard earned money on this.",
        "Defective, unreliable, and completely useless. Customer support refused to assist. Worst purchase ever.",
        "Disastrous quality. Broke within hours of opening the package. Horrible value and zero support.",
        "Total rip off! Dreadful craftsmanship, terrible materials, and completely non-functional. 1 star is too generous."
    ]
}

CATEGORY_KEYWORDS = {
    "Electronics": ["battery life", "gadget", "screen display", "connectivity", "sound clarity", "speed"],
    "Clothing": ["fabric quality", "stitching", "fit and comfort", "style", "soft material", "breathability"],
    "Books": ["writing style", "engaging chapters", "content quality", "illustrations", "storytelling", "insightful read"],
    "Home & Kitchen": ["durability", "kitchen utility", "easy cleanup", "sturdy build", "modern design", "countertop fit"],
    "Beauty": ["skin texture", "gentle formula", "fragrance", "clean application", "radiant look", "soothing effect"],
    "Sports": ["grip comfort", "athletic performance", "rugged build", "ergonomic design", "workout gear", "endurance"]
}


def generate_catalog():
    """Generate a master catalog of products with realistic prices."""
    products = []
    products_per_cat = NUM_PRODUCTS // len(CATEGORIES)
    prod_counter = 1001

    for category in CATEGORIES:
        min_p, max_p = PRICE_RANGES[category]
        for _ in range(products_per_cat):
            prod_id = f"PROD_{prod_counter}"
            # Log-uniform or beta distribution for realistic price clustering
            price = round(random.uniform(min_p, max_p), 2)
            products.append({
                "product_id": prod_id,
                "product_category": category,
                "price": price
            })
            prod_counter += 1
    return pd.DataFrame(products)


def generate_dataset():
    """Generate 50,000 product reviews with zero data leakage."""
    print("=" * 70)
    print("STEP 1: Generating Product Review Dataset (50,000 records)")
    print("=" * 70)

    # Catalog creation
    product_catalog = generate_catalog()
    products_list = product_catalog.to_dict(orient="records")

    # Generate customer list
    customer_ids = [f"CUST_{i:05d}" for i in range(10001, 10001 + NUM_CUSTOMERS)]

    # Assign reviews to customers to simulate sequential transaction history
    # Power-law / Pareto style distribution of review counts per customer
    # Some customers review once, power users review multiple times
    weights = np.random.pareto(a=1.8, size=NUM_CUSTOMERS) + 1.0
    weights /= weights.sum()
    customer_review_allocations = np.random.multinomial(TOTAL_RECORDS, weights)

    # Ensure every customer has at least 1 review if possible
    # and adjust to match exactly 50,000
    records = []

    # Rating distribution probabilities (realistic e-commerce breakdown)
    # 5 stars: 42%, 4 stars: 24%, 3 stars: 14%, 2 stars: 10%, 1 star: 10%
    rating_choices = [1, 2, 3, 4, 5]
    rating_probs = [0.10, 0.10, 0.14, 0.24, 0.42]

    # Customer state tracker: customer_id -> list of prior ratings
    # STRICT ZERO DATA LEAKAGE: Review count and avg rating use ONLY past reviews!
    customer_history = {cid: [] for cid in customer_ids}

    # Interleave reviews across customers chronologically
    # Build list of (customer_id, transaction_index)
    all_customer_events = []
    for cid, count in zip(customer_ids, customer_review_allocations):
        for seq in range(count):
            all_customer_events.append(cid)

    # Shuffle events randomly to simulate chronological arrival of reviews
    random.shuffle(all_customer_events)

    print(f"Generating review records with zero-leakage historical calculation...")

    for i, cid in enumerate(all_customer_events):
        # 1. Strictly prior history for this customer
        prior_ratings = customer_history[cid]
        customer_review_count = len(prior_ratings)
        customer_avg_rating = round(float(np.mean(prior_ratings)), 2) if customer_review_count > 0 else 0.0

        # 2. Select product
        product = random.choice(products_list)
        prod_id = product["product_id"]
        category = product["product_category"]
        price = product["price"]

        # 3. Generate current rating
        # High customer satisfaction history slightly influences subsequent satisfaction
        if customer_review_count > 0 and customer_avg_rating >= 4.0:
            probs = [0.06, 0.08, 0.12, 0.26, 0.48]
        elif customer_review_count > 0 and customer_avg_rating < 3.0:
            probs = [0.18, 0.16, 0.18, 0.22, 0.26]
        else:
            probs = rating_probs

        current_rating = int(np.random.choice(rating_choices, p=probs))

        # 4. Verified purchase status (~78% Yes, ~22% No)
        verified_purchase = "Yes" if random.random() < 0.78 else "No"

        # 5. Synthesize review text correlated with rating and category
        base_template = random.choice(REVIEW_TEMPLATES[current_rating])
        keyword = random.choice(CATEGORY_KEYWORDS[category])
        
        # Add contextual category keyword sentence naturally
        if current_rating >= 4:
            context_phrase = f" The {keyword} especially stood out during regular use."
        elif current_rating == 3:
            context_phrase = f" In terms of {keyword}, it performed as expected."
        else:
            context_phrase = f" Particularly, the {keyword} was very underwhelming."

        review_text = f"{base_template}{context_phrase}"

        # 6. Append record
        records.append({
            "product_id": prod_id,
            "product_category": category,
            "price": price,
            "review_text": review_text,
            "verified_purchase": verified_purchase,
            "customer_id": cid,
            "customer_review_count": customer_review_count,
            "customer_avg_rating": customer_avg_rating,
            "rating": current_rating
        })

        # 7. Update customer history for future reviews (after current record is finalized)
        customer_history[cid].append(current_rating)

    df = pd.DataFrame(records)

    # Ensure output directories exist
    target_paths = [
        os.path.join("data", "product_reviews.csv"),
        os.path.join("Product_Rating_Prediction", "data", "product_reviews.csv")
    ]

    for p in target_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False)
        print(f"Dataset successfully saved to: {os.path.abspath(p)}")

    # 8. Data Validation and Summary Display
    print("\n" + "=" * 70)
    print("DATASET VALIDATION & SUMMARY")
    print("=" * 70)
    print(f"Total Records Generated: {len(df):,}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Unique Customers: {df['customer_id'].nunique():,}")
    print(f"Unique Products: {df['product_id'].nunique():,}")
    
    print("\nMissing Values Summary:")
    missing = df.isnull().sum()
    print(missing)
    assert missing.sum() == 0, "Error: Missing values detected in generated dataset!"

    print("\nRating Distribution:")
    dist = df["rating"].value_counts().sort_index()
    pct = (dist / len(df) * 100).round(2)
    dist_df = pd.DataFrame({"Count": dist, "Percentage (%)": pct})
    print(dist_df)

    print("\nTarget Label Distribution (4-5 = High Rating, 1-3 = Low Rating):")
    high_count = (df["rating"] >= 4).sum()
    low_count = (df["rating"] <= 3).sum()
    print(f"High Rating (1): {high_count:,} ({high_count / len(df) * 100:.2f}%)")
    print(f"Low Rating (0):  {low_count:,} ({low_count / len(df) * 100:.2f}%)")

    print("\nCategory Summary:")
    cat_summary = df.groupby("product_category").agg(
        Review_Count=("rating", "count"),
        Avg_Price=("price", "mean"),
        Avg_Rating=("rating", "mean")
    ).round(2)
    print(cat_summary)

    print("\nSample Preview (First 5 Records):")
    print(df.head())
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    generate_dataset()

