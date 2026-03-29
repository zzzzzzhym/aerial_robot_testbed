# Note

## Parameter Fitting
Original BEMT paper brutally solve the parameter fitting by placing the entire BEMT process in parameter searching routine. 
NeuroBEM use a smarter way -- use measured thrust to determine inflow velocity (v_i), then directly use this v_i and candidate parameters to calculate the predicted thrust, and finally match predicted thrust and data sample. It avoids the inner loop to do root finding. 
So even when both of the solver use Nelder-Mead, NeuroBEM can be much faster.

In general, when optimizing params involves an inner loop for root finding, we can also break the inner root finding and consider that in the outer optimization process.