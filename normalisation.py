import vedo
import os
import matplotlib.pyplot as plt
import numpy as np

import barycenter as bc
import pcaFunctions


# find the largest bound of the mesh
def findLargestBound(mesh_: vedo.mesh.Mesh):
    bounds = mesh_.bounds()
    xbounds = abs(bounds[0]-bounds[1])
    ybounds = abs(bounds[2]-bounds[3])
    zbounds = abs(bounds[4]-bounds[5])
    # print(f"bound sizes: {xbounds}, {ybounds}, {zbounds}")
    return max(xbounds, ybounds, zbounds)


# Scale the mesh by first finding the largest bound and then computing the scaling factor
def scale(mesh_: vedo.mesh.Mesh):
    largest = findLargestBound(mesh_)
    mesh_.scale(1/largest)


# (OLD)Decimating the mesh, to around the desired number of 'desiredNumber'
def subAndSuperSampleOld(mesh_: vedo.mesh.Mesh, desiredNumber: int, boundaries: bool = False, meth: str = 'pro'):
    points = mesh_.polydata().GetNumberOfPoints()
    print(f"Points: {points}")
    if points > desiredNumber:
        fractie = desiredNumber / points
        print(f"Fraction is: {fractie}")
        mesh_.decimate(fraction=fractie, method=meth, boundaries=boundaries)


# Sub and super sample the given mesh to around the desiredNumber. If decimation using pro results in a mesh with
# number of vertices < desiredNumber + margin: don't decimate any further
def subAndSuperSample(mesh_: vedo.mesh.Mesh, desiredNumber: int, margin: int = 160):
    points = mesh_.polydata().GetNumberOfPoints()
    # print(f"Points: {points}")
    if points > desiredNumber + margin:
        fractie = desiredNumber / points
        # print(f"Fraction is: {fractie}")
        mesh_.decimate(fraction=fractie, method='pro')
        points = mesh_.polydata().GetNumberOfPoints()
        # print(f"Points now: {points}")
        if points > desiredNumber + margin:
            fractie = desiredNumber / points
            mesh_.decimate(fraction=fractie, method='quadric')


# Translating the mesh to barycenter-origin collision
def translationNormalisation(mesh_: vedo.mesh.Mesh, xyz: list = []):
    if len(xyz) == 3:
        x, y, z = xyz[0], xyz[1], xyz[2]
    else:
        x, y, z = bc.findBaryCenter2(mesh_)
    print(f"The barycenter: {x}, {y}, {z}")
    mesh_.pos(-x, -y, -z)


# Allign the mesh using PCA
def allignMesh(mesh_: vedo.mesh.Mesh):
    new_vertices = pcaFunctions.pca(mesh_, True)
    new_mesh = mesh_.points(new_vertices)


# Plotting for histograms
def makeHistogram(data, title: str = 'Histogram of cosine similarity', xlabel: str = 'Cosine similarity', ylabel: str = 'Count',
                  binsCount: int = 10, printMinMaxAvg: bool = True, withCounts: bool = True, xlim: list = [], bins_: list = []):
    density, bins, _ = plt.hist(data, bins=binsCount) if len(bins_) == 0 else plt.hist(data, bins=bins_)
    count, _ = np.histogram(data, bins)
    print(f"The bins: {bins}")
    if withCounts:
        binsDiff = bins[1] - bins[0]
        for x, y, num in zip(bins, density, count):
            if num != 0:
                print(f"add text {num} <> {x}, {y}")
                # plt.text(x + (0.02 * max(bins)), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str
                plt.text(x + (0.2 * binsDiff), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if printMinMaxAvg:
        print(f"The average is: {np.average(data)}, min: {min(data)} & max: {max(data)}")
    if len(xlim) == 2:
        print(f"The xlim min: {xlim[0]} & max: {xlim[1]}")
        plt.xlim(xlim[0], xlim[1])
    plt.show()
    return bins


# Plot the histogram
def makeHistogram2(data, title: str = 'Histogram of cosine similarity', xlabel: str = 'Cosine similarity', ylabel: str = 'Count',
                  binsCount: int = 10, printMinMaxAvg: bool = True, withCounts: bool = True, xlim: list = [],
                   bins: list = [], fileName: str = "", bbox_inches: str = 'tight', printNoShow: bool = False):
    density, bins, _ = plt.hist(data, bins=binsCount, weights=np.ones(len(data))/len(data)) if len(bins) == 0 else plt.hist(data, bins=bins)
    count, _ = np.histogram(data, bins)
    print(f"The bins: {bins}")
    if withCounts:
        binsDiff = bins[1] - bins[0]
        for x, y, num in zip(bins, density, count):
            if num != 0:
                print(f"add text {num} <> {x}, {y}")
                # plt.text(x + (0.02 * max(bins)), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str
                plt.text(x + (0.08 * binsDiff), y + (1.08 * max(density)/100), "{:.4f}".format(num/sum(count))[1:], fontsize=10, rotation=0)  # x,y,str

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if printMinMaxAvg:
        print(f"The average is: {np.average(data)}, min: {min(data)} & max: {max(data)}")
    if len(xlim) == 2:
        print(f"The xlim min: {xlim[0]} & max: {xlim[1]}")
        plt.xlim(xlim[0], xlim[1])
    if printNoShow:
        plt.savefig(f'figures/{fileName}.png', bbox_inches=bbox_inches) if len(fileName) > 1 else \
            plt.savefig(f'figures/{title}.png', bbox_inches=bbox_inches)
    else:
        plt.show()


# Normalise the data using decimation
def remeshDatabase1(write: bool = True):
    ans = []
    subfolders = [f.path for f in os.scandir('DB') if f.is_dir()]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if file.endswith(".off") or file.endswith(".ply"):
                print(f"File {file}")
                mesh_ = vedo.load(subfolder + '/' + file)
                subAndSuperSample(mesh_, 1500, margin=2)
                ans.append(mesh_.polydata().GetNumberOfPoints())

                if write:
                    subff = subfolder.split('\\')[1]
                    fileNew = file.split('.')[0]+".ply"
                    vedo.write(mesh_, f'DB_decim/{subff}/{fileNew}', binary=False)
    # Do stuff with histogram
    makeHistogram(ans, 'Histogram of number of vertices', 'Number of vertices')


# Translate normalisation for the database
def remeshDatabase2(database: str, write: bool = True, folderName: str = 'trans'):
    bef = []
    ans = []
    subfolders = [f.path for f in os.scandir(database) if f.is_dir()]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if file.endswith(".off") or file.endswith(".ply"):
                print(f"File {file}")
                mesh_ = vedo.load(subfolder + '/' + file)

                x, y, z = bc.findBaryCenter2(mesh_)
                dist = bc.calcLength(np.array([0.0, 0.0, 0.0]), np.array([x, y, z]))
                bef.append(dist)
                translationNormalisation(mesh_, [x, y, z])

                x2, y2, z2 = bc.findBaryCenter2(mesh_)
                dist = bc.calcLength(np.array([0.0, 0.0, 0.0]), np.array([x2, y2, z2]))
                ans.append(dist)

                if write:
                    subff = subfolder.split('\\')[1]
                    fileNew = file.split('.')[0]+".ply"
                    vedo.write(mesh_, f'DB_{folderName}/{subff}/{fileNew}', binary=False)
    # Do stuff with histogram
    maxW = max(bef)
    bins = np.arange(0.0, maxW+0.00001, (maxW/10))
    print("Histogram before translation")
    makeHistogram(bef, 'Histogram of distance to origin', 'Distance center-origin', bins_=bins)
    print("Print important stuff")
    print(bef)
    print(ans)
    print(bins)
    print("Histogram after translation")
    makeHistogram(ans, 'Histogram of distance to origin', 'Distance center-origin', bins_=bins)


# PCA: normalisation of the allignment for the database
def remeshDatabase3(database: str, write: bool = True, folderName: str = 'pca'):
    bef = []
    ans = []
    subfolders = [f.path for f in os.scandir(database) if f.is_dir()]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if file.endswith(".off") or file.endswith(".ply"):
                print(f"File {file}")
                mesh_ = vedo.load(subfolder + '/' + file)

                translationNormalisation(mesh_)
                e1, e2 = pcaFunctions.pca(mesh_, False)
                cssSum = pcaFunctions.calculateCosineSimilarityIndentitySummed(e1, e2)
                bef.append(cssSum)
                new_vertices = pcaFunctions.pca(mesh_, True)
                new_mesh = mesh_.points(new_vertices)
                e1_, e2_ = pcaFunctions.pca(new_mesh, False)
                cssSum2 = pcaFunctions.calculateCosineSimilarityIndentitySummed(e1_, e2_)
                ans.append(cssSum2)

                if write:
                    subff = subfolder.split('\\')[1]
                    fileNew = file.split('.')[0]+".ply"
                    vedo.write(mesh_, f'DB_{folderName}/{subff}/{fileNew}', binary=False)
    # Do stuff with histogram
    maxW = max(bef)
    bins = np.arange(0.0, 2.00001, (2.000001/10))
    print("Histogram before pca")
    makeHistogram(bef, 'Histogram of cosine similarity summed', 'Summed cosine similarity', bins_=bins)
    print("Print important stuff")
    print(list(bins))
    print(bef)
    print(ans)
    print("Histogram after pca")
    makeHistogram(ans, 'Histogram of cosine similarity summed', 'Summed cosine similarity', xlim=[0, 2.0], bins_=bins)


# Scaling the meshes in the database
def remeshDatabase4(database: str, write: bool = True, folderName: str = 'scale'):
    bef = []
    ans = []
    subfolders = [f.path for f in os.scandir(database) if f.is_dir()]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if file.endswith(".off") or file.endswith(".ply"):
                print(f"File {file} in {subfolder}")
                mesh_ = vedo.load(subfolder + '/' + file)

                maxB = findLargestBound(mesh_)
                bef.append(maxB)
                scale(mesh_)

                maxB2 = findLargestBound(mesh_)
                ans.append(maxB2)

                if write:
                    subff = subfolder.split('\\')[1]
                    fileNew = file.split('.')[0] + ".ply"
                    vedo.write(mesh_, f'DB_{folderName}/{subff}/{fileNew}', binary=False)
    # Do stuff with histogram
    print("Histogram before scaling")
    makeHistogram(bef, 'Histogram of max bound', 'Max bound')
    print("Histogram after scaling")
    makeHistogram(ans, 'Histogram of max bound', 'Max bound', bins=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])


#
#
# Below for testing only
#
#


def mainTesT():
    # remeshDatabase1()
    # remeshDatabase2("DB_decim", write=False)
    remeshDatabase3("DB_trans", write=False, folderName='pca')
    # remeshDatabase4("DB_pca", write=False)
    # mesh_ = vedo.load('DB_trans/Airplane/61.ply')
    # print(f"Faces: {len(mesh_.faces())} & points: {mesh_.polydata().GetNumberOfPoints()} & vertices: {len(mesh_.vertices())}")
    # mesh_.show(axes=8)


def printDist():
    bins = [0.0, 0.200001, 0.400002, 0.6000030000000001, 0.800004, 1.000005, 1.2000060000000001, 1.400007, 1.600008, 1.8000090000000002, 2.0000001]
    bef = [0.004707974655072961, 0.012055686513783265, 0.021470977719988613, 0.00017645954088392422, 1.9996777940973822,
     1.9989701451512132, 1.9972487637998968, 0.15284766640651817, 1.9714362924402429, 0.005129255761546831,
     0.07041230076043484, 0.00902836645407256, 0.0357450423117823, 0.03574504262130836, 0.013442772956835422,
     1.9991400524626344, 1.9967219872167792, 0.007320036187243455, 1.9922846591517647, 0.8375959976046592,
     0.02805442792881275, 0.36998220821419414, 0.08516846017716413, 0.059939800759244294, 0.10761546210840249,
     0.03508680202573038, 0.47572021523714514, 0.09774944904909015, 0.22454466669366702, 0.10113985885771823,
     0.10512031298838262, 0.04110365453789387, 0.005236609569510027, 0.007917339483461145, 0.13863353389795435,
     1.1975419250190595, 0.39542832960491636, 0.8252243480470985, 0.06546900662756364, 0.07721307448763191,
     0.05047522651620194, 0.698841098119137, 0.17138430085153838, 0.0841552574758378, 0.4293265719943041,
     0.40963822361682767, 0.3461981539529499, 0.020987267951763408, 1.001295478490005, 1.0902891895614744,
     0.4275104563948441, 0.22570575968795012, 0.26194869034999035, 0.30040085649473475, 0.03527932946611771,
     0.11328210597844425, 0.26476166826042624, 1.3921854391546211, 0.04829990381049594, 0.8617255178672723,
     1.7671212203685647, 0.010157036671500688, 0.03308941598666233, 0.1160061018025918, 0.01859575510101632,
     0.0098908226013555, 0.01641870705495668, 0.8521720141419926, 0.008375113869763303, 0.007852763316166236,
     0.2798082277005975, 0.8215780642301258, 0.6289327830245437, 1.8077907070876265, 0.004712659711332908,
     0.005227095801115926, 0.004084488436028449, 1.0181697738270807, 0.011003016925197853, 0.00455563551179365,
     0.05320344911421871, 0.03038229550545603, 1.405299252863559, 1.724116641594268, 1.0329283543288963,
     1.4532840554194784, 1.5123181733122784, 1.4261917663592456, 1.8489665445341368, 1.1900446415806674,
     0.057088792003620416, 1.724116641594268, 1.6028028444498421, 1.1581707476767629, 0.0005449412163377337,
     1.0738088457654715, 0.12176049875385783, 1.0004977161362676, 0.8905245455242962, 1.1121644871224905,
     1.3020182132470828, 0.14699393476045058, 0.015137473679053751, 0.6985771954719799, 1.9988114238814056,
     1.1367470479572068, 0.20291798126263288, 1.4299763383106536, 0.6080782554582356, 1.0751955202327652,
     1.9981730934103643, 0.30159717307935, 1.9058986301612584, 0.41889425968218463, 0.031967802773039185,
     0.2697067938658352, 0.31316424837856577, 0.08457593611380215, 0.4600070214904578, 0.9642339561534222,
     0.08334527690722741, 0.43141725523497887, 0.33705529295141506, 0.35588652544277327, 0.18170962857683598,
     0.17235594654308717, 1.5944866594637945, 0.17973679685984711, 0.16906165599110878, 0.04850293445430513,
     0.02910334167906275, 0.04284380074275999, 0.008700402855537269, 0.42839548488086565, 0.30788085080922173,
     0.39271360367504937, 0.024401022211361485, 0.028546552006905444, 0.37290650185353746, 0.0782847889397357,
     0.025148135706260222, 1.999559997394143, 1.7746626497256486, 1.0611639943943894, 1.7319010292293227,
     0.022420339654914297, 0.24731880711247434, 0.41596441101863135, 1.9996471838757783, 0.4770387164633929,
     0.05115398288838266, 1.1203076606077027, 0.9022522380655329, 1.9961049054212026, 0.9856606818802761,
     0.024422889946783793, 0.10957761779474777, 0.43957921264007305, 1.9925495844700956, 1.7156885709527148,
     0.013326904850925917, 0.9926076577229851, 1.017963143603055, 1.972235793994924, 1.9777315104308357,
     1.0495955039676632, 1.0542146946246502, 1.0542146946246502, 0.033300056151083224, 1.0040786185981028,
     1.979982785213898, 1.973602794223696, 1.9670388488196315, 1.9266095890945705, 1.9758153922745905,
     1.925921044911843, 1.9580217626991114, 1.977643028883798, 1.9487279623110283, 1.9825295168122217,
     1.9479581087476174, 1.8302555380852135, 0.6479842033346326, 0.07344487481845322, 0.13909136387159426,
     1.754149380301941, 1.5745580427030865, 1.539649324646331, 1.914575093144254, 0.5796108338867156,
     0.37606193353798545, 0.9736737192727171, 1.9949314451451738, 0.6175815089293724, 1.0438741872415098,
     1.8312785557254774, 1.7486513578797198, 0.9219440320385266, 1.582339671496365, 0.5067759159654122,
     0.972414708105458, 0.12741118923823772, 1.9038160601487728, 0.9822597184371096, 0.1416598102216468,
     1.0073930724279825, 1.9562887568201304, 1.0394272369837816, 1.9830709478027835, 1.9952460937832304,
     0.14555376489185592, 1.9995568229762635, 1.9970160367729777, 1.9515949372328822, 0.3847261401098878,
     1.0205663359088881, 0.22839034961305807, 1.0475623961962022, 1.3395107486391589, 1.997847705147267,
     1.9498801366178506, 0.44799712609021686, 0.29223429198451517, 1.0898180249378715, 1.1535502808537488,
     0.3576094919693751, 0.28723432653854425, 0.3803029856320584, 0.30611879758218125, 1.2522733566912647,
     0.5989196334173085, 0.4993806999963001, 0.3462513430258315, 0.242253782860882, 0.08299753392446281,
     0.05636818755513405, 1.1966089068440489, 0.10194469160282192, 0.22909799352077445, 0.030017716204268647,
     0.5186379199405042, 0.007873825075976596, 0.005005205922689498, 0.0014060754201509018, 0.012529405519498452,
     0.01854874438863083, 0.7288748738965984, 0.08488573996405659, 0.25350371945763495, 0.2628996173844738,
     0.31586548723236796, 0.0597583309188651, 0.031056657552191048, 0.018685533316152816, 0.048914601161905,
     0.23033869634982113, 0.04142179450539118, 0.014602067550384193, 1.0030780528122603, 0.6896813862641649,
     0.6316937609169163, 0.05895376453702658, 0.21922489278549911, 0.06929665028866137, 0.9001068115435263,
     0.5397661609127369, 1.9482593824940393, 1.8959857202613937, 0.12017709798884911, 0.21404084378160337,
     0.4266085219373138, 0.791827666974785, 1.1943684215176675, 0.7230774968869017, 0.11383515070738634,
     0.2676634615196951, 0.061619999196264055, 0.47527952816415764, 1.1578483321283088, 0.7020635915139493,
     0.12147501805284794, 1.8908538584928811, 1.993497338488456, 1.8240490718471323, 0.9082108878494636,
     1.2195819455055301, 0.15923508041563836, 1.272553370297485, 0.42229776628035964, 1.837439923587616,
     0.6886798122765767, 0.6977431142651344, 1.0024453098949941, 1.1563608796727043, 0.4079328296734217,
     0.3190785085360973, 1.5129876308675632, 1.4696822006689385, 1.9792516829347662, 1.9839608913084121,
     1.4481331000193527, 0.3495486014958673, 0.3540697897863054, 0.20516460611470674, 0.9687028378610827,
     0.2834394039363156, 0.19542770069179924, 0.21981908176255247, 0.27742964137216897, 0.9948561745985652,
     0.9936396007434997, 0.9770174204196695, 0.9261668387783901, 0.4990128174483511, 0.4233283727767668,
     0.3926584667999566, 0.35121451031893003, 0.17005274173117457, 0.165690251419322, 0.8259061830387014,
     0.8887710639873295, 1.997459172440231, 1.0101070649643464, 1.0081362637399203, 1.0016761359109612,
     0.0888790104961374, 0.9245371395953643, 0.6284672959125132, 0.051313301622054844, 0.09452238933749477,
     0.04791445462389314, 0.7805180265689784, 0.09414690821677994, 0.031241378203189077, 1.9642648388793145,
     1.0517846892915408, 0.8616689700335899, 1.002994526951853, 0.6304952013339029, 0.41045679123510637,
     0.23807392983740555, 0.13861112179781337, 0.24016490002660665, 0.03872618374789635, 0.17670916168715098,
     0.08940107690288511, 0.1371195776906359, 0.0917210984102734, 0.159864820885335, 0.049478497504597,
     0.19517540956445656, 0.02957509002818625, 0.237554384205834, 0.04518872960615761, 0.08829101272968155,
     0.22314823254926008, 0.2944183099983477, 0.09590321917206426, 0.04156693519840703, 0.359151581069174,
     0.010509796892576629, 0.05434401016428016, 0.16681629919990687, 0.055433797808444416, 1.9931270918860684,
     1.163775585971906, 0.14157741340653532, 1.9999777863132433, 0.4915290052899527, 0.28026622383397803,
     0.036782024269565186, 0.06134246247066234, 1.9593484222862414, 0.08345121302916608, 0.07534454265024515,
     1.0682689374448853, 1.1268496731589726, 0.026467809235868698, 1.9506100850570443, 1.9953352625112708,
     0.041367217189255207]
    data = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999996,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     1.9999999999999978, 1.9999999999999991, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999933, 2.0, 2.0,
     2.0, 2.0, 1.9999999999999996, 1.9999999999999996, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 1.999999999999997, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     1.9999999999999996, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999991, 2.0, 2.0, 2.0, 1.9999999999999996, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     1.9999999999999996, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 1.9999999999999993, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999996, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999996, 2.0, 2.0, 2.0, 2.0,
     2.0, 2.0, 2.0, 2.0, 1.9999999999999998, 1.9999999999999991, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    # makeHistogram(data, 'Histogram of distance to origin', 'Distance center-origin', bins_=bins)
    density, bins, _ = plt.hist(data, bins=bins)
    count, _ = np.histogram(data, bins)
    print(f"The bins: {bins}")
    binsDiff = bins[1] - bins[0]
    for x, y, num in zip(bins, density, count):
        if num != 0:
            print(f"add text {num} <> {x}, {y}")
            # plt.text(x + (0.02 * max(bins)), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str
            plt.text(x + (0.2 * binsDiff), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str
    plt.title('Histogram of cosine similarity summed')
    plt.xlabel( 'Summed cosine similarity')
    plt.ylabel('Count')
    print(f"The average is: {np.average(data)}, min: {min(data)} & max: {max(data)}")
    # if len(xlim) == 2:
    #     print(f"The xlim min: {xlim[0]} & max: {xlim[1]}")
    plt.xlim(0, 2.00001)
    plt.show()

if __name__ == '__main__':
    printDist()
    # mainTesT()
# remeshDatabase4('DB', False)