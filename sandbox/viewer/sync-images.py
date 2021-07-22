#!/usr/bin/python
import json
import os, fnmatch
import logging
import fcntl
from concurrent.futures import ProcessPoolExecutor, Future, Executor
import csv
import argparse
import datetime
import pprint

from typing import TextIO, List

EXPERIMENTROOT = "/home/matejg/Project/crc_ml/models/evaluations/"
IMAGESDIR = "test/figs"
IGNOREDIRS = [ '.gitignore', '.git', '.gitkeep']
EXPERIMENTINFO = "summary.json"

IIMSERVERROOT = "/mnt/data/IIPImage/"
IIMSERVERIMGSRC = "srcimg"
IIMSERVERANNOT = "annotation"
IIMSERVEREXPERIMENTS = "experiments"

IMGSRCDIRS = [ "/mnt/data/scans/ADOPT scans", "/mnt/data/scans/AI scans", "/mnt/data/scans/Biobanka" ]
IMGSRCEXTS = [ "*.mrxs" ]

ANNOTDIRS = [ "/home/matejg/Project/crc_ml/data/interim/Prostata/level1/masks/annotations", "/home/matejg/Project/crc_ml/data/interim/Mammy/level1/masks/annotations" ]
ANNOTEXTS = [ "*.png" ]

IIPMOOSRV = "http://glados9.cerit-sc.cz:8080/iipmooviewer/index-dev.html"
IIPMOOROOT = "/mnt/data/iipmooviewer/"
IIPMOOHTML = "list-experiments.html"
IIPMOOCSVEXPCONFIG = "experiments-config.csv"
IIPMOOCSVEXPRESULTS = "experiments-results.csv"
CSVSEPARATOR = ";"

CONVERTTOTIFF = "vips tiffsave"
CONVERTTOTIFFPARAMSOVERLAY = "--tile --pyramid --compression=deflate --tile-width 256 --tile-height 256 --bigtiff"
CONVERTTOTIFFPARAMSSRC = "--tile --pyramid --compression=jpeg --Q=80 --tile-width 256 --tile-height 256 --bigtiff"

LOCKFILE = '/tmp/sync-images-lock'
NUMPARALLELPROCS = 24

# Parse command line to modify constants

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose information on progress of the processing')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='debug information on progress of the processing')
parser.add_argument('-p', '--parallel', dest='parallel', nargs=1, help='number of parallel processes for VIPS processing of overlays')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
elif args.verbose:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.ERROR)


if args.parallel is not None:
    NUMPARALLELPROCS = int(args.parallel[0])

# starting actual work

logging.debug("Acquiring lock")
lock = open(LOCKFILE, 'w')  # type: TextIO
fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
logging.debug("Lock acquired")

html =  """
<html>
<body>
<style>
/* Tooltip container */
.tooltip {
  position: relative;
  display: inline-block;
  border-bottom: 1px dotted black; /* If you want dots under the hoverable text */
}

/* Tooltip text */
.tooltip .tooltiptext {
  visibility: hidden;
  width: 120px;
  background-color: black;
  color: #fff;
  text-align: center;
  padding: 5px 0;
  border-radius: 6px;

  /* Position the tooltip text - see examples below! */
  position: absolute;
  z-index: 1;
}

/* Show the tooltip text when you mouse over the tooltip container */
.tooltip:hover .tooltiptext {
  visibility: visible;
}
</style>

<h1>Experiments</h1>
"""

html += '<p><a href="' + IIPMOOCSVEXPCONFIG + '">Experiment config CSV</a></p>' + "\n"
html += '<p><a href="' + IIPMOOCSVEXPRESULTS + '">Experiment results CSV</a></p>' + "\n"

html += """
<ul>
"""

def indexDirs (dirsToScan : List[str], validExtensions : List[str]) -> (dict, dict):
    index = {}
    bannedIndex = {}
    for sourceDir in dirsToScan:
        for root, dirs, files in os.walk(sourceDir, topdown=False):
            for name in files:
                bname = os.path.basename(name)
                bnameWithoutExt = os.path.splitext(bname)[0]
                if any(fnmatch.fnmatch(bname, ext) for ext in validExtensions) and not bnameWithoutExt in bannedIndex:
                    if bnameWithoutExt in index:
                        logging.error("Duplicate file found: " + bnameWithoutExt + "\nPrevious location: " + index[bnameWithoutExt] + "\nCurrent location: " + os.path.join(root, name))
                        del index[bnameWithoutExt]
                        bannedIndex[bnameWithoutExt] = True
                        logging.info("Banned " + bnameWithoutExt + " from indexing!")
                    index[bnameWithoutExt] = os.path.join(root, name)
                    logging.info("Indexed file " + bnameWithoutExt + " as " + os.path.join(root, name))
    return index, bannedIndex

sourceImageIndex, sourceImageIndexBanned = indexDirs(IMGSRCDIRS, IMGSRCEXTS)
annotationImageIndex, annotationImageIndexBanned = indexDirs(ANNOTDIRS, ANNOTEXTS)

def forkVips(arg : str) -> int :
    return os.system(arg)

def formatStats(results : dict) -> str:
    stats = []
    try:
        if 'metrics' in results:
            if 'accuracy' in results['metrics']:
                stats.append("A=" + '%.2f' % (float(results['metrics']['accuracy'])))
            if 'precision' in results['metrics']:
                stats.append("P=" + '%.2f' % (float(results['metrics']['precision'])))
            if 'recall' in results['metrics']:
                stats.append("R=" + '%.2f' % (float(results['metrics']['recall'])))
            if 'loss' in results['metrics']:
                stats.append("L=" + '%.2f' % (float(results['metrics']['loss'])))
            if 'auc' in results['metrics']:
                stats.append("AUC=" + '%.2f' % (float(results['metrics']['auc'])))
            if stats:
                return " (" + ",".join(stats) + ")"
        return ""
    except KeyError as e:
        return " (unable to parse results)"

def processImage(sourceFile : str, targetDir : str, targetFile : str, conversionParams : str, pool : Executor) -> Future :
    if not os.path.exists(sourceFile):
        raise FileNotFoundError("Source file for conversion is missing: " + sourceFile)
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    if not os.path.exists(os.path.join(targetDir, targetFile)):
        logging.info("   converting " + os.path.join(targetDir, targetFile))
        assert visFileWithPath is not None
        callParams = CONVERTTOTIFF + ' "' + sourceFile + '" "' + os.path.join(targetDir, targetFile)  + '" ' + conversionParams
        return pool.submit(forkVips, callParams)
    else:
        logging.info("   skipping already converted image " + os.path.join(targetDir, targetFile))
        return None

futures = []
experimentDate = {}
experimentHTML = {}
experimentOverlays = {}
experimentResults = {}
experimentMetadata = {}
experimentMetadataHeader = []
with ProcessPoolExecutor(NUMPARALLELPROCS) as pool:
    for experiment in os.listdir(EXPERIMENTROOT):
        # skip ignored files/dirs without even initializing anything
        if experiment in IGNOREDIRS:
            continue
        experimentOverlays[experiment] = {}
        experimentResults[experiment] = {}
        logging.info("Checking experiment " + experiment)
        imagesDir = os.path.join(EXPERIMENTROOT,experiment,IMAGESDIR)
        if not os.path.exists(imagesDir):
            logging.info("   no image directory found")
            continue
        try:
            metadataFileName = os.path.join(EXPERIMENTROOT, experiment, EXPERIMENTINFO)
            metadataFile = open(metadataFileName, "r")
            metadata = json.load(metadataFile)
            metadataFile.close()
            metadataFileDate = datetime.datetime.strptime(metadata['time_start'], "%d-%b-%Y %H:%M:%S")
            experimentDate[experiment] = metadataFileDate
            experimentMetadata[experiment] = metadata
        except (OSError, IOError) as e:
            metadata = None
            logging.warning("   no experiment metadata found for experiment " + experiment + ", skipping")
            continue
        except json.decoder.JSONDecodeError as e:
            metadata = None
            logging.warning("   unable to parse metadata for experiment " + experiment + ", skipping")
            continue

        srcImages = []
        for bnameWithoutExt in experimentMetadata[experiment]['test']['summary']:
            visFileWithPath = None
            logging.debug("   processing " + bnameWithoutExt)
            try:
                visFileWithPath = experimentMetadata[experiment]['test']['summary'][bnameWithoutExt]['figures']['heatmap']
                visFile = os.path.basename(visFileWithPath)
                visFile = os.path.splitext(visFile)[0]
                if visFile != bnameWithoutExt + "-vis":
                    logging.error("Unexpected naming structure of the heatmap file: should read " + bnameWithoutExt + "-vis" + " actually reads " + visFile)
                    continue
                experimentOverlays[experiment][visFile] = True
            except:
                logging.warning("   no heatmap image found for " + experiment + " figure " + bnameWithoutExt)
                continue

            assert visFileWithPath is not None
            f = processImage(visFileWithPath, os.path.join(IIMSERVERROOT, IIMSERVEREXPERIMENTS, experiment), bnameWithoutExt + '-vis.tif', CONVERTTOTIFFPARAMSOVERLAY, pool)
            if f is not None: futures.append(f) # this runs in parallel and is picked up later

            localFutures = [] # this is for futures that are local to the experiment

            # now we also handle the source image if needed
            f = processImage(sourceImageIndex[bnameWithoutExt], os.path.join(IIMSERVERROOT, IIMSERVERIMGSRC), bnameWithoutExt + '.tif', CONVERTTOTIFFPARAMSSRC, pool)
            if f is not None: localFutures.append(f)

            # now we also handle the annotation visualizations
            f = processImage(annotationImageIndex[bnameWithoutExt], os.path.join(IIMSERVERROOT, IIMSERVERANNOT), bnameWithoutExt + '-annot.tif', CONVERTTOTIFFPARAMSOVERLAY, pool)
            if f is not None: localFutures.append(f)

            for f in localFutures:
                f.result() # we get results within this experiment

            srcImages.append(bnameWithoutExt)

        if srcImages:
            experimentHTMLElement = '<li><strong>' + experiment + "</strong>\n" + '<ul>'
            experimentHTMLElement += '<li>' + metadataFileDate.strftime("%Y-%m-%d %H:%M")
            experimentHTMLElement += '<li> ' + experimentMetadata[experiment]['description'] + ' </li>' + "\n"
            experimentHTMLElement += '<li> <i>params:</i> ' + pprint.pformat(experimentMetadata[experiment]['params']) + '</li>' + "\n"
            experimentHTMLElement += '<li> <i>training:</i> ' + pprint.pformat(experimentMetadata[experiment]['train']) + '</li>' + "\n"
            experimentHTMLElement += '<li> <i>validation:</i> ' + formatStats(experimentMetadata[experiment]['validate']['summary']) + '</li>' + "\n"

            # we deal with images for which we have numeric results but they are not in visualization
            firstImage = True
            missingImagesNonEmpty = False
            for srcImage in experimentMetadata[experiment]['test']['summary']:
                if srcImage in srcImages:
                    continue
                missingImagesNonEmpty = True
                if firstImage:
                    experimentHTMLElement += '<li>missing images with results: '
                else:
                    experimentHTMLElement += ', '
                firstImage = False
                experimentHTMLElement += srcImage
                stats = formatStats(experimentMetadata[experiment]['test']['summary'][srcImage])
                experimentHTMLElement += stats
            if missingImagesNonEmpty:
                experimentHTMLElement += '</li>'

            # now we print links with images
            experimentHTMLElement += '<li>'
            firstImage = True
            for srcImage in srcImages:
                missingOverlays = ""
                if not srcImage in annotationImageIndex:
                    missingOverlays += " (missing annotation!)"
                if not srcImage + "-vis" in experimentOverlays[experiment]:
                    missingOverlays += " (missing probabilities!)"
                comma = ""
                if not firstImage:
                    comma = " | "
                stats = ""
                if srcImage in experimentMetadata[experiment]['test']['summary']:
                    stats = formatStats(experimentMetadata[experiment]['test']['summary'][srcImage])
                experimentHTMLElement += comma + '<a href= "' + IIPMOOSRV + '?image=' + os.path.join(IIMSERVERIMGSRC,
                                                                                                     srcImage) + '.tif&annotation=' + os.path.join(
                    IIMSERVERANNOT, srcImage) + '-annot.tif&probabilities=' + os.path.join(
                    IIMSERVEREXPERIMENTS, experiment,
                    srcImage) + '-vis.tif' + '">' + srcImage + '</a>' + missingOverlays + stats
                firstImage = False
            experimentHTMLElement += '</li> </ul>' + "\n"
            experimentHTML[experiment] = experimentHTMLElement
        else:
            logging.info("   no images found")
for f in futures:
    f.result()

logging.debug("Adding HTML for experiments ")
for experiment, date in sorted(experimentDate.items(), key=lambda x: x[1], reverse=True):
    if experiment in experimentHTML:
        logging.debug("Adding HTML piece for experiment " + experiment)
        html += experimentHTML[experiment]

html += """
</ul>
</body>
</html>
"""
htmlOut = open(os.path.join(IIPMOOROOT,IIPMOOHTML), "w")
htmlOut.write(html)
htmlOut.close()

logging.info("Generating CSV with experiment config")
with open(os.path.join(IIPMOOROOT,IIPMOOCSVEXPCONFIG), mode='w') as csvExpConfig:
    csvExpConfigWriter = csv.writer(csvExpConfig, delimiter=CSVSEPARATOR, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvExpConfigWriter.writerow(['experiment UUID', 'date'] + experimentMetadataHeader)
    for experiment in experimentMetadata:
        date = experimentDate[experiment].strftime("%Y-%m-%d %H:%M")
        row = [experiment, date]
        for col in experimentMetadataHeader:
            if col in experimentMetadata[experiment]:
                row.append(experimentMetadata[experiment][col])
            else:
                row.append('')
        csvExpConfigWriter.writerow(row)

logging.info("Generating CSV with experiment results")
with open(os.path.join(IIPMOOROOT,IIPMOOCSVEXPRESULTS), mode='w') as csvExpResults:
    csvExpResultsWriter = csv.writer(csvExpResults, delimiter=CSVSEPARATOR, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvExpResultHeaders = ['accuracy', 'precision', 'recall', 'auc']
    csvExpResultsWriter.writerow(['experiment UUID', 'srcImage'] + csvExpResultHeaders)
    for experiment in experimentResults:
        for srcImage in experimentResults[experiment]:
            row = [experiment, srcImage]
            for e in csvExpResultHeaders:
                if e in experimentResults[experiment][srcImage]:
                    logging.debug("Experiment result: " + experiment +  " " + srcImage + " " + e + "=" + experimentResults[experiment][srcImage][e])
                    row.append(experimentResults[experiment][srcImage][e])
                else:
                    row.append('')
            logging.debug("This is what we are going to write: " + CSVSEPARATOR.join(row))
            csvExpResultsWriter.writerow(row)

logging.debug("Releasing lock")
lock.close()
logging.debug("Lock released")
