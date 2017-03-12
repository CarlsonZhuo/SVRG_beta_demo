function [ funv ] = Eval_loss( X, y, w, loss_model, lambda )

Xw 		= X'*w;

if strcmp(loss_model, 'logistic');
    funv = sum(log(1 + exp(-y' * Xw))) 	+ 0.5*lambda*norm(w,2)^2;
else strcmp(loss_model, 'leastsq');
    funv = 0.5*(Xw - y)'*(Xw - y)		+ 0.5*lambda*norm(w,2)^2;
end


end
