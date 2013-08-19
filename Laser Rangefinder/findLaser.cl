__kernel void getLaserCoord(__read_only image2d_t srcImg,
                            __write_only image2d_t dstImg,
                            sampler_t sampler,
                            int width, int height,
                            __global int *sumX, __global int *N, __global int *sumY)
        {
          //int xpos = get_group_id(0) * get_local_size(0) + get_local_id(0); //always 0
          //int ypos = get_group_id(1) * get_local_size(1) + get_local_id(1);
          
          int xpos = get_global_id(0);
          int ypos = get_global_id(1);
          
          if (xpos >= width || ypos >= height) return; //boundary checks

          int2 outImageCoord = (int2) (xpos, ypos);
  
          if (read_imagef(srcImg, sampler, outImageCoord).x > 0.90F &&
              read_imagef(srcImg, sampler, outImageCoord).y > 0.90F &&
              read_imagef(srcImg, sampler, outImageCoord).z > 0.90F) //check threshold, x = R; y = G; z = B; w? = A = 255; between 0-1 in float4
          {
              atomic_add(N, 1);       
              atomic_add(sumX, xpos);
              atomic_add(sumY, ypos);
          }
          else
          {
              float4 s = {0,0,0,0};
              write_imagef(dstImg, outImageCoord, s);
          }
          
          /*atomic_add(N, 1);       
          atomic_add(sumX, xpos);
          atomic_add(sumY, ypos);*/
              
          //float4 h = {0.50F, 0.50F, 0.50F, 0.50F};
          //write_imagef(dstImg, outImageCoord, (X*(read_imagef(srcImg, sampler, outImageCoord)-h)+h));
          //write_imagef(dstImg, outImageCoord, read_imagef(srcImg, sampler, outImageCoord));
        }