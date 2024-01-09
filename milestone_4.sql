SELECT * FROM lp_m4;

--what percentage of the loans are recovered against the investor funding and the total amount funded?
SELECT  total_payment,
        total_payment_inv,
        funded_amount,
        funded_amount_inv,
        out_prncp,
        out_prncp_inv
FROM lp_m4;
