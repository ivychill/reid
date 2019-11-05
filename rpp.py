
stage = 'RPP'
model = model.convert_to_rpp()
if use_gpu:
    model = model.cuda()
model = rpp_train(model, criterion, f, stage, 5)

############################
# step4: whole net training #
stage = 'full'

full_train(model, criterion, f, stage, 10)


