# TXOs-prediction
在配置好程序运行所依赖的环境后，按照以下顺序运行顺序：
1. 首先运行molecule create-smile.py，共设置了10种噻吨酮类的的分子核心结构以及23种常见取代基，衍生化方法采用对称取代方式，因此最终生成的衍生分子具有（1）无取代基；（2）单取代；（3）对称位点相同取代基双取代；（4）对称位点不同取代基双取代 这四类分子结构，并以SMILES字符串形式表示，经过以上衍生方式共可得到2760个噻吨酮类衍生分子，存储于TXs.csv中。
2. 从该2760个噻吨酮类衍生分子中随机挑选276个（10%）在PBE1PBE-D3(BJ)/6-31G(d)水平、n,n-DiMethylAcetamide隐式溶剂模型下依次进行第一三重激发态的结构优化，以及在第一三重激发态最低能量结构下，使用PBE1PBE-D3(BJ)/6-31G(d)水平、n,n-DiMethylAcetamide隐式溶剂模型的单重态单点能计算，所得数据存储于database.csv中，其中energy(Ha，Triplet）指三重态分子能量，单位为Hartree，energy（Ha，Singlet）指三重态结构下单重态单点能能量，单位为Hartree，energy指三重态结构下垂直发射能，单位为kcal/mol，homo（Ha）指α-HOMO轨道能量，单位为Hartree，homo指α-HOMO轨道能量，单位为eV，lumo（Ha）指β-LUMO轨道能量，单位为Hartree，lumo指β-LUMO轨道，单位为eV。【如果分子中含有I，则I使用SDD基组与SDD赝势】
3. RF、GIN、DNN分别为使用随机森林模型、图同构神经网络和深度神经网络对该database.csv进行学习的机器学习算法，并用于预测TXs.csv中的分子性质
4. RF运行方法：直接运行py程序
5. DNN运行方法：直接运行py程序
6. GIN运行方法：首先运行run_main-bayes opy.py，然后运行run_kfold.py（可选）
7. 在DNN模型和GIN模型中，加入了使用streamlit实现的可交互筛选，命名为app.py，该筛选平台可分别使用优化得到的最佳模型进行噻吨酮类衍生物分子性质预测、分子库中分子性质的筛选与探索功能【使用该功能前务必先运行4/5/6步】。
