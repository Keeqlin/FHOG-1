#include "fhog.hpp"
#include <assert.h>
#include <float.h>
#include <string.h>
#include <opencv2/imgproc.hpp>
#include <iostream>


// 9 bins, 10 boundaries
// boundary_x: cos(theta)
static float boundary_x[] = {	 1.000000000, 0.939692616,  0.766044438, 
                                 0.499999970, 0.173648104, -0.173648298,
                                -0.500000060,-0.766044617, -0.939692676, 
                                -1.000000000};
// boundary_x: sin(theta)
static float boundary_y[] = {	 0.000000000, 0.342020154,  0.642787635,
                                 0.866025448, 0.984807789,	0.984807730,
                                 0.866025448, 0.642787457,  0.342020005,
                                 0.000000000};

int FHOG::static_Init(cv::Size _sz, int _cellSize)
{
    if (_sz == imageSize && cellSize == _cellSize)
    {
        return FHOG_OK;
    }
    imageSize = _sz;
    cellSize = _cellSize;

    // why need to substract 2 ???
    sz = cv::Size(imageSize.width  / _cellSize,
                  imageSize.height / _cellSize);
    
    element_num = sz.width * sz.height;
    // insensitive (9) + conttast sensitive(9*2) +
    // 4 dimensions capturing the overall gradient energy in square blocks of four cells around (i,j)
    numFeatures = NUM_SECTOR * 3 + 4;
	map = cv::Mat::zeros(element_num, numFeatures, CV_32FC1);

	originalFeature   = cv::Mat::zeros(element_num, ORIGINAL_FEATURE_NUM,   CV_32FC1);
	normalizedFeature = cv::Mat::zeros(element_num, NORMALIZED_FEATURE_NUM, CV_32FC1);

    // record gradient information
	alpha = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32SC2);
	r     = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32FC1);

    // auxiliary helper: precalculated terms of interploation
    // index of near cell
    nearest.create(1, cellSize, CV_32SC1);
    // voting/interpolation ratio 
    w.create(1, 2*cellSize, CV_32FC1);
	

    for(int i = 0; i < cellSize / 2; ++i)
    {
        nearest.at<int>(0, i) = -1;
    }
    for(int i = cellSize / 2; i < cellSize; i++)
    {
        nearest.at<int>(0, i) = 1;
    }

 
    for(int j = 0; j < cellSize / 2; j++)
    {
        float temp = cellSize / 2 + j + 0.5f;
        w.at<float>(0, j * 2)= 1.0f * temp / (float)cellSize; 
        w.at<float>(0, j * 2 + 1) = 1 - w.at<float>(0, j * 2);  
    }
    for(int j = cellSize / 2; j < cellSize; j++)
    {
        float temp = cellSize - (j - cellSize / 2 + 0.5f);
        w.at<float>(0, j * 2)= 1.0f * temp / (float)cellSize; 
        w.at<float>(0, j * 2 + 1) = 1 - w.at<float>(0, j * 2);  
    }
    partOfNorm.create(sz.height, sz.width, CV_32FC1);

    return FHOG_OK;
}

int FHOG::compute(const cv::Mat &src, cv::Mat &dst, int _cellSize, float thres)
{
    static_Init(src.size(), _cellSize);

    if(FHOG_OK != getFeatureMaps(src))
    {
        return FHOG_ERR_FEATUREMAP;
    }
    if(FHOG_OK != normalizeAndTruncate(thres))
    {
        return FHOG_ERR_NORM;
    }
    if(FHOG_OK != PCAFeatureMaps())
    {
        return FHOG_ERR_PCA;
    }
    dst = map;
    return FHOG_OK;
}

int FHOG::getFeatureMaps(const cv::Mat &src)
{
    int height = src.rows;
    int width  = src.cols;
	int numChannels = src.channels();

	float kernel[3] = { -1.f, 0.f, 1.f };
	cv::Mat kernel_dx(1, 3, CV_32F, kernel);
	cv::Mat kernel_dy(3, 1, CV_32F, kernel);

	cv::Mat dx;
	cv::Mat dy;

	cv::filter2D(src, dx, CV_16S, kernel_dx, cv::Point(-1,0), 0.0, cv::BORDER_REFLECT);
	cv::filter2D(src, dy, CV_16S, kernel_dy, cv::Point(0,-1), 0.0, cv::BORDER_REFLECT);

    for (int i = 1; i < height - 1; ++i)
    {
        // int16_t *pdx = dx.ptr<int16_t>(i);
        // int16_t *pdy = dy.ptr<int16_t>(i);
        float *pr = r.ptr<float>(i);
        cv::Vec2i *palpha = alpha.ptr<cv::Vec2i>(i);
		size_t row_index = width * i;
        for (int j = 1; j < width - 1; ++j)
        {
			size_t pixel_index_ = numChannels*(row_index + j);
			// gradient magnitude of pixel(j,i)
            int x = 0;
            int y = 0;
			pr[j] = 0;
			for (int ch = 0; ch <numChannels; ch++) {
				// frame.data[frame.channels()*(frame.cols*y + x) + 0];  channel 1
				int16_t tx = ((int16_t*)dx.data)[pixel_index_ + ch];
				int16_t ty = ((int16_t*)dy.data)[pixel_index_ + ch];
				float mag = sqrt(tx * tx + ty * ty);
				if ( mag > pr[j]) {
					pr[j] = mag;
					x = tx;
					y = ty;
				}
			}
        
            float dp1 = fabs(boundary_x[1] * x + boundary_y[1] * y);
			float dp2 = fabs(boundary_x[4] * x + boundary_y[4] * y);
			float dp3 = fabs(boundary_x[7] * x + boundary_y[7] * y);

            int maxi = (dp1 > dp2 && dp1 > dp3) ? 1 : dp2 > dp3 ? 4 : 7;
            int maxii = maxi - 1;
            float max = boundary_x[maxii] * x + boundary_y[maxii] * y;
            
            if (max < 0)
            {
                max = -max;
                maxii += NUM_SECTOR;
            }

            for (int kk = maxi; kk <= maxi+1; kk+=1)
            {
                float dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
                if(fabs(dotProd) > max)
                {
                    max = fabs(dotProd);
                    maxii = kk + (dotProd < 0 ? NUM_SECTOR : 0);
                }
            }
            // bin-index of contrast insensitive 
            palpha[j][0] = maxii % NUM_SECTOR;
            // bin-index of contrast sensitive 
            palpha[j][1] = maxii;
        }
    }


    // Bilinear interpolation
    int sizeX      = sz.width;
    int sizeY      = sz.height;
    int stringSize = 3 * sizeX * NUM_SECTOR;

    originalFeature.setTo(0);
    // auxiliary helper
    float *pw   = w.ptr<float>();
    int *pn     = nearest.ptr<int>();
    float *pm   = originalFeature.ptr<float>();
    float *pr   = r.ptr<float>();
    int *palpha = alpha.ptr<int>();


    for(int i = 0; i < sizeY; ++i)
	{
		for(int j = 0; j < sizeX; ++j)
		{
			for(int ii = 0; ii < cellSize; ++ii)
			{
				for(int jj = 0; jj < cellSize; ++jj)
				{
                    // shifted cell index of originalFeature
					int i_nii = i + pn[ii];
					int j_njj = j + pn[jj];
					int i_niixss = i_nii * stringSize;
					int j_njjxmn = j_njj * 3 * NUM_SECTOR;
                    
                    // current cell index of originalFeature
					int ixss = i * stringSize;
					int jxmn = j * 3 * NUM_SECTOR;
                    
                    int iix2 = ii * 2;
					int jjx2_0 = jj * 2;
					int jjx2_1 = jjx2_0 + 1;
					
                    // pixel index
                    int ixk_ii = i * cellSize + ii;
					int jxk_jj = j * cellSize + jj;
					if ((ixk_ii > 0) && (ixk_ii < height - 1) && (jxk_jj > 0) && (jxk_jj < width  - 1))
					{
						int d = (ixk_ii) * width + (jxk_jj);
						int dx2 = d * 2;
                        // bin-index of orientation
                        int alfa_dx2_0   = palpha[dx2];                  // insenstive(9): 0 - 8
						int alfa_dx2_1_N = palpha[dx2 + 1] + NUM_SECTOR; // sensitive(18): 9 - 26 
                        
                        // pr[d] is gradient magnitude
                        float rdxwii2_1 = pr[d] * pw[iix2 + 1];
						float rdxwii2_0 = pr[d] * pw[iix2];

						float rw00 = rdxwii2_0 * pw[jjx2_0];
						float rw11 = rdxwii2_1 * pw[jjx2_1];
						float rw10 = rdxwii2_1 * pw[jjx2_0];
						float rw01 = rdxwii2_0 * pw[jjx2_1];

						pm[ ixss + jxmn + alfa_dx2_0  ] += rw00;
						pm[ ixss + jxmn + alfa_dx2_1_N] += rw00;

						if ((i_nii >= 0) && (i_nii <= sizeY - 1) && (j_njj >= 0) && (j_njj <= sizeX - 1))
						{
							pm[i_niixss + j_njjxmn + alfa_dx2_0  ] += rw11;
							pm[i_niixss + j_njjxmn + alfa_dx2_1_N] += rw11;

							pm[i_niixss + jxmn 	   + alfa_dx2_0  ] += rw10;
							pm[i_niixss + jxmn 	   + alfa_dx2_1_N] += rw10;
							pm[ixss     + j_njjxmn + alfa_dx2_0  ] += rw01;
							pm[ixss     + j_njjxmn + alfa_dx2_1_N] += rw01;
						}
						else if ((i_nii >= 0) && (i_nii <= sizeY - 1))
						{
							pm[i_niixss + jxmn 	+ alfa_dx2_0  ] += rw10;
							pm[i_niixss + jxmn 	+ alfa_dx2_1_N] += rw10;
						}
						else if ((j_njj >= 0) && (j_njj <= sizeX - 1))
						{
							pm[ixss + j_njjxmn + alfa_dx2_0  ] += rw01;
							pm[ixss + j_njjxmn + alfa_dx2_1_N] += rw01;
						}
					}
				}
			}
		}
	}

    return FHOG_OK;
}

int FHOG::normalizeAndTruncate(float thres)
{

    float* pPartOfNorm = partOfNorm.ptr<float>();
    // originalFeature.rows = sz.width * sz.height
    for(int i = 0; i < originalFeature.rows; ++i)
    {
        float valOfNorm = 0.0f;
        float* pm = originalFeature.ptr<float>(i);
        // only sum the squre of contrast insensitive value
        for(int j = 0; j < NUM_SECTOR; j++)
        {
        	float mm_p_j = pm[j];
            valOfNorm += (mm_p_j * mm_p_j);
        }
        pPartOfNorm[i] = valOfNorm;
    }

    // normalization
    for(int i = 1; i < sz.height-1; ++i)
    {
        float* pPartOfNorm_curr = partOfNorm.ptr<float>(i);
        float* pPartOfNorm_last = partOfNorm.ptr<float>(i-1);
        float* pPartOfNorm_next = partOfNorm.ptr<float>(i+1);

        size_t index = i * (sz.width);
        for(int j = 1; j < sz.width-1; ++j)
        {
            float pN_0 = pPartOfNorm_curr[j    ];
            float pN_1 = pPartOfNorm_curr[j + 1];
            float pN_2 = pPartOfNorm_next[j    ];
            float pN_3 = pPartOfNorm_next[j + 1];
            float pN_4 = pPartOfNorm_last[j    ];
            float pN_5 = pPartOfNorm_last[j + 1];
            float pN_6 = pPartOfNorm_curr[j - 1];
            float pN_7 = pPartOfNorm_next[j - 1];
            float pN_8 = pPartOfNorm_last[j - 1];
            float valOfNorm1 = 1.f / sqrt(pN_0 + pN_1 + pN_2 + pN_3 + FLT_EPSILON);
            float valOfNorm2 = 1.f / sqrt(pN_0 + pN_1 + pN_4 + pN_5 + FLT_EPSILON);
            float valOfNorm3 = 1.f / sqrt(pN_0 + pN_6 + pN_2 + pN_7 + FLT_EPSILON);
            float valOfNorm4 = 1.f / sqrt(pN_0 + pN_6 + pN_4 + pN_8 + FLT_EPSILON);

            float* pOriginalFeature   = originalFeature.ptr<float>(index + j);
            float* pNormalizedFeature = normalizedFeature.ptr<float>(index + j);
            for(int ii = 0; ii < NUM_SECTOR; ii++)
            {
                float mm_idx1 = pOriginalFeature[ii];
                pNormalizedFeature[ii] = mm_idx1 * valOfNorm1;
                pNormalizedFeature[ii + NUM_SECTOR] = mm_idx1 * valOfNorm2;
                pNormalizedFeature[ii + NUM_SECTOR * 2] = mm_idx1 * valOfNorm3;
                pNormalizedFeature[ii + NUM_SECTOR * 3] = mm_idx1 * valOfNorm4;

            }
            pOriginalFeature = pOriginalFeature + NUM_SECTOR;
            for(int ii = 0; ii < 2 * NUM_SECTOR; ii++)
            {
                float mm_idx1 = pOriginalFeature[ii];
                pNormalizedFeature[ii + NUM_SECTOR * 4] = mm_idx1 * valOfNorm1;
                pNormalizedFeature[ii + NUM_SECTOR * 6] = mm_idx1 * valOfNorm2;
                pNormalizedFeature[ii + NUM_SECTOR * 8] = mm_idx1 * valOfNorm3;
                pNormalizedFeature[ii + NUM_SECTOR * 10]= mm_idx1 * valOfNorm4;
            }
        }
    }

    // Truncate
    float* pNormalizedFeature = normalizedFeature.ptr<float>();
    for(int i = 0; i < normalizedFeature.rows * normalizedFeature.cols; ++i)
    {
        if (pNormalizedFeature [i] > thres) {
            pNormalizedFeature[i] = thres;
        }
    }
    
    return FHOG_OK;
}


int FHOG::PCAFeatureMaps()
{
    float nx    = 1.f / sqrt((float)(NUM_SECTOR * 2)+FLT_EPSILON);
    float ny    = 0.5f; 
    // float ny    = 1.f / sqrt((float)(4)+FLT_EPSILON);


    for(int i = 0; i < sz.height; i++)
    {
        for(int j = 0; j < sz.width; j++)
        {
            float* pNormalizeFeature = normalizedFeature.ptr<float>(i*sz.width+j);
            float* pMap = map.ptr<float>(i*sz.width+j);
            memset(pMap, 0, sizeof(float) * numFeatures);

            // contrast sensitive orientation
            for(int m = 0; m < NUM_SECTOR * 2; ++m)
            {
                float* phead = pNormalizeFeature + 4 * NUM_SECTOR + m;
                for(int n = 0; n < 4; ++n)
                {
                    pMap[m] += *phead;
                    phead   += 2 * NUM_SECTOR;
                }
                pMap[m] *= ny;
            }

            // contrast insensitive orientation
            pMap += 2 * NUM_SECTOR;
            for(int m = 0; m < NUM_SECTOR; ++m)
            {
                float* phead = pNormalizeFeature + m;
                for(int n = 0; n < 4; ++n)
                {
                    pMap[m] += *phead;
                    phead += NUM_SECTOR;
                }
                pMap[m] *= ny;
            }

            // 4 dimensions capturing the overall gradient energy
            pMap += NUM_SECTOR;
            for(int m = 0; m < 4; ++m)
            {
                float* phead = pNormalizeFeature + m*NUM_SECTOR;
                for(int n = 0; n < NUM_SECTOR; ++n)
                {
                    pMap[m] += *(phead++);
                }
                phead = pNormalizeFeature + (2*m+4)*NUM_SECTOR;
                for(int n = 0; n < 2 * NUM_SECTOR; ++n)
                {
                    pMap[m] += *(phead++);
                }
                pMap[m] *= nx;
            }
        }
    }

    return FHOG_OK;
}
