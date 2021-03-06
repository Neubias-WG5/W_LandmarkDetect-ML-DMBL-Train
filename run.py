# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *	  http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"

from ldmtools import *
import numpy as np
from multiprocessing import Pool
import scipy.ndimage as snd
from sklearn.externals import joblib
import sys, os
from neubiaswg5 import CLASS_LNDDET
from neubiaswg5.helpers import NeubiasJob, prepare_data, get_discipline
from cytomine.models import Job, AttachedFile, Property
import joblib
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
import imageio

def image_dataset_phase_1(repository, image_number, x, y, feature_offsets, R_offsets, delta, P):
	img = makesize(snd.zoom(readimage(repository, image_number, image_type='tif'), delta), 1)

	(h, w) = img.shape
	mask = np.ones((h, w), 'bool')
	mask[:, 0] = 0
	mask[0, :] = 0
	mask[h - 1, :] = 0
	mask[:, w - 1] = 0
	(nroff, blc) = R_offsets.shape
	h -= 2
	w -= 2
	x += 1
	y += 1

	n_out = int(np.round(P * nroff))
	rep = np.zeros((x.size * nroff) + n_out)
	xs = np.zeros((x.size * nroff) + n_out).astype('int')
	ys = np.zeros((x.size * nroff) + n_out).astype('int')
	for ip in range(x.size):
		xs[ip * nroff:(ip + 1) * nroff] = x[ip] + R_offsets[:, 0]
		ys[ip * nroff:(ip + 1) * nroff] = y[ip] + R_offsets[:, 1]
		rep[ip * nroff:(ip + 1) * nroff] = ip
	mask[ys, xs] = 0
	(ym, xm) = np.where(mask == 1)
	perm = np.random.permutation(ym.size)[0:n_out]
	ym = ym[perm]
	xm = xm[perm]
	xs[x.size * nroff:] = xm
	ys[y.size * nroff:] = ym
	rep[x.size * nroff:] = x.size
	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep


def dataset_mp_helper(jobargs):
	return image_dataset_phase_1(*jobargs)


def get_dataset_phase_1(repository, image_ids, n_jobs, feature_offsets, R_offsets, delta, P, X, Y):
	p = Pool(n_jobs)
	Xc = np.round(X * delta).astype('int')
	Yc = np.round(Y * delta).astype('int')
	(nims, nldms) = Xc.shape
	jobargs = []

	for i in range(nims):
		jobargs.append((repository, image_ids[i], Xc[i, :], Yc[i, :], feature_offsets, R_offsets, delta, P))
	data = p.map(dataset_mp_helper, jobargs)
	p.close()
	p.join()
	(nroff, blc) = R_offsets.shape
	n_in = nroff * nldms
	n_out = int(np.round(nroff * P))
	n_tot = n_in + n_out
	DATASET = np.zeros((nims * n_tot, feature_offsets[:, 0].size))
	REP = np.zeros(nims * n_tot)
	IMG = np.zeros(nims * n_tot)
	b = 0
	i = 0
	for (d, r) in data:
		(nd, nw) = d.shape
		DATASET[b:b + nd, :] = d
		REP[b:b + nd] = r
		IMG[b:b + nd] = i
		i += 1
		b = b + nd
	DATASET = DATASET[0:b, :]
	REP = REP[0:b]
	IMG = IMG[0:b]
	return DATASET, REP, IMG


def build_phase_1_model(repository, image_ids=[], n_jobs=1, NT=32, F=100, R=2, sigma=10, delta=0.25, P=1, X=None, Y=None):
	std_matrix = np.eye(2) * (sigma ** 2)
	feature_offsets = np.round(np.random.multivariate_normal([0, 0], std_matrix, NT * F)).astype('int')
	R_offsets = []

	for x1 in range(-R, R + 1):
		for x2 in range(-R, R + 1):
			if (np.linalg.norm([x1, x2]) <= R):
				R_offsets.append([x1, x2])

	R_offsets = np.array(R_offsets).astype('int')
	(dataset, rep, img) = get_dataset_phase_1(repository, image_ids, n_jobs, feature_offsets, R_offsets, delta, P, X, Y)
	return dataset, rep, img, feature_offsets

def image_dataset_phase_2(repository, image_number, x, y, feature_offsets, R_offsets, delta):
	img = makesize(snd.zoom(readimage(repository, image_number, image_type='tif'), delta), 1)
	(h, w) = img.shape
	mask = np.ones((h, w), 'bool')
	mask[:, 0] = 0
	mask[0, :] = 0
	mask[h - 1, :] = 0
	mask[:, w - 1] = 0
	(nroff, blc) = R_offsets.shape
	h -= 2
	w -= 2
	x += 1
	y += 1
	rep = np.zeros((nroff, 2))
	number = image_number
	xs = (x + R_offsets[:, 0]).astype('int')
	ys = (y + R_offsets[:, 1]).astype('int')
	rep[:, 0] = R_offsets[:, 0]
	rep[:, 1] = R_offsets[:, 1]
	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep, number


def dataset_mp_helper_phase_2(jobargs):
	return image_dataset_phase_2(*jobargs)


def get_dataset_phase_2(repository, image_ids, n_jobs, feature_offsets, R_offsets, delta, Xc, Yc):
	p = Pool(n_jobs)
	nims = Xc.size
	jobargs = []
	for i in range(nims):
		jobargs.append((repository, image_ids[i], Xc[i], Yc[i], feature_offsets, R_offsets, delta))

	data = p.map(dataset_mp_helper_phase_2, jobargs)
	p.close()
	p.join()
	(nroff, blc) = R_offsets.shape
	nims = len(image_ids)
	DATASET = np.zeros((nims * nroff, feature_offsets[:, 0].size))
	REP = np.zeros((nims * nroff, 2))
	NUMBER = np.zeros(nims * nroff)
	b = 0

	for (d, r, n) in data:
		(nd, nw) = d.shape
		DATASET[b:b + nd, :] = d
		REP[b:b + nd, :] = r
		NUMBER[b:b + nd] = n
		b = b + nd

	DATASET = DATASET[0:b, :]
	REP = REP[0:b]
	NUMBER = NUMBER[0:b]
	return DATASET, REP, NUMBER


def build_phase_2_model(repository, image_ids=None, n_jobs=1, NT=32, F=100, R=3, N=500, sigma=10, delta=0.25, Xc = None, Yc = None):
	std_matrix = np.eye(2) * (sigma ** 2)
	feature_offsets = np.round(np.random.multivariate_normal([0, 0], std_matrix, NT * F)).astype('int')
	R_offsets = np.zeros((N, 2))
	dis = np.random.ranf(N) * R
	ang = np.random.ranf(N) * 2 * np.pi
	R_offsets[:, 0] = np.round((dis * np.cos(ang))).astype('int')
	R_offsets[:, 1] = np.round((dis * np.sin(ang))).astype('int')
	(dataset, rep, number) = get_dataset_phase_2(repository, image_ids, n_jobs, feature_offsets, R_offsets, delta, Xc, Yc)
	return dataset, rep, number, feature_offsets


def build_edgematrix_phase_3(Xc, Yc, sde, delta, T):
	Xc = Xc * delta
	Yc = Yc * delta
	(nims, nldms) = Xc.shape
	differential_entropy = np.eye(nldms) + np.inf
	c1 = np.zeros((nims, 2))
	c2 = np.zeros((nims, 2))

	for ldm1 in range(nldms):
		c1[:, 0] = Xc[:, ldm1]
		c1[:, 1] = Yc[:, ldm1]
		for ldm2 in range(ldm1 + 1, nldms):
			c2[:, 0] = Xc[:, ldm2]
			c2[:, 1] = Yc[:, ldm2]
			diff = c1 - c2
			d = diff - np.mean(diff, axis=0)
			d = np.mean(np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2)))
			differential_entropy[ldm1, ldm2] = d
			differential_entropy[ldm2, ldm1] = d

	edges = np.zeros((nldms, T))

	for ldm in range(nldms):
		edges[ldm, :] = np.argsort(differential_entropy[ldm, :])[0:T]

	return edges.astype(int)

def	get_neubias_coords(gt_path, tr_im):
	first_im = imageio.imread(os.path.join(gt_path, '%d.tif'%tr_im[0]))
	nldms = np.max(first_im)
	nimages = len(tr_im)
	xcs = np.zeros((nimages, nldms))
	ycs = np.zeros((nimages, nldms))
	xrs = np.zeros((nimages, nldms))
	yrs = np.zeros((nimages, nldms))
	for i in range(len(tr_im)):
		id = tr_im[i]
		gt_img = imageio.imread(os.path.join(gt_path, '%d.tif'%id))
		for id_term in range(1, nldms+1):
			(y, x) = np.where(gt_img==id_term)
			(h, w) = gt_img.shape
			yc = y[0]
			xc = x[0]
			yr = yc/h
			xr = xc/w
			xcs[i, id_term-1] = xc
			ycs[i, id_term-1] = yc
			xrs[i, id_term-1] = xr
			yrs[i, id_term-1] = yr
	return np.array(xcs), np.array(ycs), np.array(xrs), np.array(yrs)

def main():
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		conn.job.update(progress=0, status=Job.RUNNING, statusComment="Initialization of the training phase...")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=True, **conn.flags)

		tmax = 1
		for f in os.listdir(gt_path):
			if f.endswith('.tif'):
				gt_img = imageio.imread(os.path.join(gt_path, f))
				tmax = np.max(gt_img)
				break

		term_list = range(1, tmax + 1)
		tr_im = [int(id_im) for id_im in conn.parameters.cytomine_training_images.split(',')]
		(xc, yc, xr, yr) = get_neubias_coords(gt_path, tr_im)
		(nims, nldms) = xc.shape
		Xc = np.zeros((nims, len(term_list)))
		Yc = np.zeros(Xc.shape)
		for id_term in term_list:
			Xc[:, id_term - 1] = xc[:, id_term - 1]
			Yc[:, id_term - 1] = yc[:, id_term - 1]
		conn.job.update(progress=10, status=Job.RUNNING, statusComment="Building model for phase 1")
		(dataset, rep, img, feature_offsets_1) = build_phase_1_model(in_path, image_ids=tr_im, n_jobs=conn.parameters.model_njobs, F=conn.parameters.model_F_P1, R=conn.parameters.model_R_P1, sigma=conn.parameters.model_sigma, delta=conn.parameters.model_delta, P=conn.parameters.model_P, X=Xc, Y=Yc)
		clf = SeparateTrees(n_estimators=int(conn.parameters.model_NT_P1), n_jobs=int(conn.parameters.model_njobs))
		clf = clf.fit(dataset, rep)
		model_filename = joblib.dump(clf, os.path.join(out_path, 'model_phase1.joblib'), compress=3)[0]
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=model_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()

		model_filename = joblib.dump((Xc, Yc), os.path.join(out_path, 'coords.joblib'), compress=3)[0]
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=model_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()

		model_filename = joblib.dump(feature_offsets_1, os.path.join(out_path, 'offsets_phase1.joblib'), compress=3)[0]
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=model_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()

		for id_term in conn.monitor(term_list, start=20, end=80, period=0.05,prefix="Visual model building for terms..."):
			(dataset, rep, number, feature_offsets_2) = build_phase_2_model(in_path, image_ids=tr_im, n_jobs=conn.parameters.model_njobs, NT=conn.parameters.model_NT_P2, F=conn.parameters.model_F_P2, R=conn.parameters.model_R_P2, N=conn.parameters.model_ns_P2, sigma=conn.parameters.model_sigma, delta=conn.parameters.model_delta, Xc = Xc[:, id_term-1], Yc = Yc[:, id_term-1])
			reg = SeparateTreesRegressor(n_estimators=int(conn.parameters.model_NT_P2), n_jobs=int(conn.parameters.model_njobs))
			reg.fit(dataset, rep)
			model_filename = joblib.dump(reg, os.path.join(out_path, 'reg_%d_phase2.joblib'%id_term), compress=3)[0]
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=model_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			model_filename = joblib.dump(feature_offsets_2, os.path.join(out_path, 'offsets_%d_phase2.joblib' % id_term), compress=3)[0]
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=model_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()

		conn.job.update(progress=90, status=Job.RUNNING, statusComment="Building model for phase 3")
		edges = build_edgematrix_phase_3(Xc, Yc, conn.parameters.model_sde, conn.parameters.model_delta, conn.parameters.model_T)
		model_filename = joblib.dump(edges, os.path.join(out_path, 'model_edges.joblib'), compress=3)[0]
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=model_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()

		sfinal = ""
		for id_term in term_list:
			sfinal += "%d " % id_term
		sfinal = sfinal.rstrip(' ')
		Property(conn.job, key="id_terms", value=sfinal.rstrip(" ")).save()
		conn.job.update(progress=100, status=Job.TERMINATED, statusComment="Job terminated.")
if __name__ == "__main__":
	main()