import torch
import torch.nn.functional as F

def BCE_loss_on_skills(predict, skill, target):

    L, N, C = predict.size()
    predict = predict.view(L * N, C)
    skill = skill.view(L * N)
    target = target.view(L * N)

    no_ignore_inds = target > -1
    predict = predict[no_ignore_inds]
    skill = skill[no_ignore_inds]
    target = target[no_ignore_inds]

    predict = predict.gather(-1, skill[:, None]).squeeze(-1)

    target = target.to(predict.dtype)

    return F.binary_cross_entropy_with_logits(predict, target)

def BCE_loss(predict, target):
    L, N, _ = predict.size()
    predict = predict.view(L * N)
    target = target.view(L * N)

    no_ignore_inds = target > -1
    predict = predict[no_ignore_inds]
    target = target[no_ignore_inds]

    target = target.to(predict.dtype)

    return F.binary_cross_entropy_with_logits(predict, target)

def BCE_loss_on_skills_for_seq(predict, skill, target):
    N, C = predict.size()
    # predict = predict.view(L * N, C)
    skill = skill.view(N)
    target = target.view(N)

    # no_ignore_inds = target > -1
    # predict = predict[no_ignore_inds]
    # skill = skill[no_ignore_inds]
    # target = target[no_ignore_inds]

    predict = predict.gather(-1, skill[:, None]).squeeze(-1)
    target = target.to(predict.dtype)
    return F.binary_cross_entropy_with_logits(predict, target)


