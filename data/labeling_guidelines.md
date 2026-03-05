# Labeling Guidelines: Multi-Label Issue Classification (Level B)

## General Rules
- Assign one or more issue labels per review.
- If no listed issue applies, set `other=1`.
- Multi-label is expected: one review can mention multiple problems.
- Keep labels tied to explicit text evidence (not assumptions).

## Label Definitions + Examples
1. `delivery_shipping`
- Shipping or delivery speed/tracking/package arrival problems.
- Example: "Arrived two weeks late and tracking never updated."
- Example: "Package was marked delivered but I never received it."

2. `redemption_activation`
- Gift card redeem/activation/code/PIN problems.
- Example: "The code says invalid when I try to redeem."
- Example: "Card was never activated at checkout."

3. `product_quality`
- Physical card quality or condition defects.
- Example: "Card arrived bent and damaged."
- Example: "Printing quality was poor and hard to read."

4. `customer_service`
- Support/helpdesk response quality or behavior.
- Example: "Customer service was rude and unhelpful."
- Example: "Support never replied to my request."

5. `refund_return`
- Refund/return/reimbursement process issues.
- Example: "They refused my refund request."
- Example: "Still waiting for money back after return."

6. `usability`
- UX/workflow friction that blocks normal use.
- Example: "Redemption flow is confusing and fails repeatedly."
- Example: "Website throws errors when applying balance."

7. `value_price`
- Price/value dissatisfaction (too expensive, not worth it).
- Example: "Overpriced for what it offers."
- Example: "Not worth the money."

8. `fraud_scam`
- Scam/fraud/unauthorized-charge/security concerns.
- Example: "Looks like a scam and balance vanished."
- Example: "Unauthorized redemption happened immediately."

9. `other`
- Issue exists but none of the above labels fit.
- Example: "Problem not covered by taxonomy categories."
- Example: "General complaint with no specific issue type."

## Consistency Checks
- Avoid contradictions: if `other=1`, other labels should usually be 0.
- Prefer specific labels over `other` when evidence is clear.
- Keep annotation decisions concise in `notes` when uncertain.
