"""
Mock SSAS cube schema used during development when a real SSAS connection
is not available. Switch to a real connection by setting USE_MOCK_CUBE=false
in the environment and providing a live SSAS endpoint.
"""

CUBE = {
    "name": "Sales",
    "caption": "Sales",
    "description": "Sales cube containing revenue, order, and margin data across customers, products, and time.",
    "metadata": {"source": "mock"},
}

DIMENSIONS = [
    {
        "name": "[Customer].[Country]",
        "unique_name": "[Customer].[Country]",
        "caption": "Country",
        "cube_name": "Sales",
        "description": "The country of the customer.",
        "aliases": ["country", "region"],
        "metadata": {"source": "mock"},
    },
    {
        "name": "[Date].[Calendar]",
        "unique_name": "[Date].[Calendar]",
        "caption": "Calendar",
        "cube_name": "Sales",
        "description": "Calendar date hierarchy with year, quarter, and month levels.",
        "aliases": ["date", "calendar", "year", "month"],
        "metadata": {"source": "mock"},
    },
    {
        "name": "[Product].[Category]",
        "unique_name": "[Product].[Category]",
        "caption": "Product Category",
        "cube_name": "Sales",
        "description": "The top-level category of the product.",
        "aliases": ["product", "category"],
        "metadata": {"source": "mock"},
    },
    {
        "name": "[Customer].[Segment]",
        "unique_name": "[Customer].[Segment]",
        "caption": "Customer Segment",
        "cube_name": "Sales",
        "description": "The market segment the customer belongs to.",
        "aliases": ["segment", "premium", "standard"],
        "metadata": {"source": "mock"},
    },
]

HIERARCHIES = {
    "[Customer].[Country]": [
        {
            "dimension": "[Customer].[Country]",
            "dimension_name": "[Customer].[Country]",
            "hierarchy": "[Customer].[Country]",
            "name": "[Customer].[Country]",
            "caption": "Country",
            "level": "[Customer].[Country].[Country]",
            "levels": [
                {
                    "name": "[Customer].[Country].[Country]",
                    "unique_name": "[Customer].[Country].[Country]",
                    "caption": "Country",
                    "level_number": 1,
                    "metadata": {"source": "mock"},
                }
            ],
            "metadata": {"source": "mock"},
        }
    ],
    "[Date].[Calendar]": [
        {
            "dimension": "[Date].[Calendar]",
            "dimension_name": "[Date].[Calendar]",
            "hierarchy": "[Date].[Calendar]",
            "name": "[Date].[Calendar]",
            "caption": "Calendar",
            "level": "[Date].[Calendar].[Month]",
            "levels": [
                {
                    "name": "[Date].[Calendar].[Year]",
                    "unique_name": "[Date].[Calendar].[Year]",
                    "caption": "Year",
                    "level_number": 1,
                    "metadata": {"source": "mock"},
                },
                {
                    "name": "[Date].[Calendar].[Quarter]",
                    "unique_name": "[Date].[Calendar].[Quarter]",
                    "caption": "Quarter",
                    "level_number": 2,
                    "metadata": {"source": "mock"},
                },
                {
                    "name": "[Date].[Calendar].[Month]",
                    "unique_name": "[Date].[Calendar].[Month]",
                    "caption": "Month",
                    "level_number": 3,
                    "metadata": {"source": "mock"},
                },
            ],
            "metadata": {"source": "mock"},
        }
    ],
    "[Product].[Category]": [
        {
            "dimension": "[Product].[Category]",
            "dimension_name": "[Product].[Category]",
            "hierarchy": "[Product].[Category]",
            "name": "[Product].[Category]",
            "caption": "Product Category",
            "level": "[Product].[Category].[Category]",
            "levels": [
                {
                    "name": "[Product].[Category].[Category]",
                    "unique_name": "[Product].[Category].[Category]",
                    "caption": "Category",
                    "level_number": 1,
                    "metadata": {"source": "mock"},
                }
            ],
            "metadata": {"source": "mock"},
        }
    ],
    "[Customer].[Segment]": [
        {
            "dimension": "[Customer].[Segment]",
            "dimension_name": "[Customer].[Segment]",
            "hierarchy": "[Customer].[Segment]",
            "name": "[Customer].[Segment]",
            "caption": "Customer Segment",
            "level": "[Customer].[Segment].[Segment]",
            "levels": [
                {
                    "name": "[Customer].[Segment].[Segment]",
                    "unique_name": "[Customer].[Segment].[Segment]",
                    "caption": "Segment",
                    "level_number": 1,
                    "metadata": {"source": "mock"},
                }
            ],
            "metadata": {"source": "mock"},
        }
    ],
}

MEASURES = [
    {
        "name": "[Measures].[Net Revenue]",
        "unique_name": "[Measures].[Net Revenue]",
        "caption": "Net Revenue",
        "cube_name": "Sales",
        "description": "Total net sales revenue after discounts.",
        "aggregation": "sum",
        "format_string": "#,##0.00",
        "aliases": ["sales", "revenue"],
        "metadata": {"source": "mock"},
    },
    {
        "name": "[Measures].[Order Count]",
        "unique_name": "[Measures].[Order Count]",
        "caption": "Order Count",
        "cube_name": "Sales",
        "description": "Total number of orders placed.",
        "aggregation": "count",
        "format_string": "#,##0",
        "aliases": ["order", "count"],
        "metadata": {"source": "mock"},
    },
    {
        "name": "[Measures].[Gross Margin]",
        "unique_name": "[Measures].[Gross Margin]",
        "caption": "Gross Margin",
        "cube_name": "Sales",
        "description": "Gross profit margin (revenue minus cost of goods sold).",
        "aggregation": "sum",
        "format_string": "#,##0.00",
        "aliases": ["margin", "gross", "kar", "kâr"],
        "metadata": {"source": "mock"},
    },
]

# Known dimension members used to guide question and MDX generation.
# Date members cover 2024-2026 since the system is operating in 2026.
MEMBERS = {
    "[Customer].[Country]": [
        {"member_unique_name": "[Customer].[Country].&[Turkey]",        "caption": "Turkey"},
        {"member_unique_name": "[Customer].[Country].&[Germany]",       "caption": "Germany"},
        {"member_unique_name": "[Customer].[Country].&[United States]", "caption": "United States"},
    ],
    "[Customer].[Segment]": [
        {"member_unique_name": "[Customer].[Segment].&[Premium]",  "caption": "Premium"},
        {"member_unique_name": "[Customer].[Segment].&[Standard]", "caption": "Standard"},
    ],
    "[Product].[Category]": [
        {"member_unique_name": "[Product].[Category].&[Bikes]",       "caption": "Bikes"},
        {"member_unique_name": "[Product].[Category].&[Accessories]", "caption": "Accessories"},
    ],
    "[Date].[Calendar].[Year]": [
        {"member_unique_name": "[Date].[Calendar].[Year].&[2024]", "caption": "2024"},
        {"member_unique_name": "[Date].[Calendar].[Year].&[2025]", "caption": "2025"},
        {"member_unique_name": "[Date].[Calendar].[Year].&[2026]", "caption": "2026"},
    ],
    "[Date].[Calendar].[Quarter]": [
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2024]&[Q1]", "caption": "2024 Q1"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2024]&[Q2]", "caption": "2024 Q2"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2024]&[Q3]", "caption": "2024 Q3"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2024]&[Q4]", "caption": "2024 Q4"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2025]&[Q1]", "caption": "2025 Q1"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2025]&[Q2]", "caption": "2025 Q2"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2025]&[Q3]", "caption": "2025 Q3"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2025]&[Q4]", "caption": "2025 Q4"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2026]&[Q1]", "caption": "2026 Q1"},
        {"member_unique_name": "[Date].[Calendar].[Quarter].&[2026]&[Q2]", "caption": "2026 Q2"},
    ],
    "[Date].[Calendar].[Month]": [
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[1]",  "caption": "January 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[2]",  "caption": "February 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[3]",  "caption": "March 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[4]",  "caption": "April 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[5]",  "caption": "May 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[6]",  "caption": "June 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[7]",  "caption": "July 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[8]",  "caption": "August 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[9]",  "caption": "September 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[10]", "caption": "October 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[11]", "caption": "November 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2024]&[12]", "caption": "December 2024"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[1]",  "caption": "January 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[2]",  "caption": "February 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[3]",  "caption": "March 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[4]",  "caption": "April 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[5]",  "caption": "May 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[6]",  "caption": "June 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[7]",  "caption": "July 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[8]",  "caption": "August 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[9]",  "caption": "September 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[10]", "caption": "October 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[11]", "caption": "November 2025"},
        {"member_unique_name": "[Date].[Calendar].[Month].&[2025]&[12]", "caption": "December 2025"},
    ],
}
